"""Block-based calibration wizard for the GPU digital twin.

Each calibration block corresponds to one physical question:
    1. White reference  -- scanner normalisation
    2. Dot sheet        -- spray footprint radius + radial profile
    3. Line sheet       -- mass deposition vs z and speed
    4. Color swatches   -- stable manifold color LUT
    5. Layering         -- multi-layer stacking validation
    6. Preview/validate -- full comparison report

Uses Rich for rendering and questionary for prompts, following the
CLI pattern established in ``robot_control/scripts/cli/``.

Usage
-----
    python -m digital_twin.calibration_cli
    python -m digital_twin.calibration_cli --config digital_twin/calibration.yaml
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import questionary
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils import color as color_utils, fs, validators
from src.utils.validators import (
    CalibrationV1,
    DotMeasurement,
    LineMeasurement,
    OverlapMeasurement,
    SwatchMeasurement,
)

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PATH = Path("digital_twin/calibration.yaml")


# ============================================================================
# Input-validation helpers
# ============================================================================

def ask_float(
    prompt: str,
    min_val: float = -1e30,
    max_val: float = 1e30,
    default: Optional[float] = None,
) -> float:
    """Prompt for a float with range validation and re-prompt on error."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = questionary.text(f"{prompt}{default_str}:").ask()
        if raw is None:
            raise KeyboardInterrupt
        raw = raw.strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
        except ValueError:
            print(f"  Invalid number: '{raw}'")
            continue
        if val < min_val or val > max_val:
            print(f"  Out of range [{min_val}, {max_val}]: {val}")
            continue
        return val


def ask_rgb(prompt: str) -> Tuple[float, float, float]:
    """Prompt for an RGB triplet (space or comma separated), each in [0,1]."""
    while True:
        raw = questionary.text(f"{prompt} (R G B in [0,1]):").ask()
        if raw is None:
            raise KeyboardInterrupt
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("  Need exactly 3 values")
            continue
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError:
            print("  Non-numeric value")
            continue
        if any(v < 0.0 or v > 1.0 for v in vals):
            print("  Each component must be in [0, 1]")
            continue
        return vals  # type: ignore[return-value]


def ask_cmy(prompt: str) -> Tuple[float, float, float]:
    """Prompt for a CMY triplet, same parsing as ``ask_rgb``."""
    while True:
        raw = questionary.text(f"{prompt} (C M Y in [0,1]):").ask()
        if raw is None:
            raise KeyboardInterrupt
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("  Need exactly 3 values")
            continue
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError:
            print("  Non-numeric value")
            continue
        if any(v < 0.0 or v > 1.0 for v in vals):
            print("  Each component must be in [0, 1]")
            continue
        return vals  # type: ignore[return-value]


def confirm_values(
    console: Console,
    label: str,
    data: Dict[str, Any],
) -> bool:
    """Show a Rich table of entered values and ask for confirmation."""
    table = Table(title=label)
    table.add_column("Field")
    table.add_column("Value")
    for k, v in data.items():
        if isinstance(v, (tuple, list)):
            v_str = ", ".join(f"{x:.4f}" for x in v)
        elif isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        table.add_row(k, v_str)
    console.print(table)
    return questionary.confirm("Accept these values?", default=True).ask()


# ============================================================================
# Calibration session
# ============================================================================

class CalibrationSession:
    """Holds the calibration model, console, and simulator reference.

    Parameters
    ----------
    calibration_path : Path
        Path to ``calibration.yaml``.
    console : Console
        Rich console for rendering.
    """

    def __init__(self, calibration_path: Path, console: Console) -> None:
        self.cal_path = calibration_path
        self.cal = validators.load_calibration_config(calibration_path)
        self.console = console
        self.modified = False
        self._sim = None

    def save(self) -> None:
        """Atomically write current calibration model to disk."""
        validators.save_calibration_config(self.cal, self.cal_path)
        self.modified = False
        self.console.print("[green]Calibration saved.[/green]")

    @property
    def simulator(self):
        """Lazy-loaded GPU simulator, rebuilt when calibration changes."""
        if self._sim is None:
            self.save()
            from digital_twin.gpu_simulator import GPUStampSimulator
            self._sim = GPUStampSimulator(self.cal_path)
        return self._sim

    def invalidate_simulator(self) -> None:
        """Force simulator rebuild on next access."""
        self._sim = None


# ============================================================================
# Block 1 -- White reference
# ============================================================================

def block_white_reference(session: CalibrationSession) -> None:
    """Measure and store the scanner paper-white RGB."""
    session.console.print(Panel(
        "Measure the paper-white RGB from the scanner.\n"
        "This normalises all subsequent measurements.",
        title="Block 1: White Reference",
    ))
    rgb = ask_rgb("Paper white RGB")
    data = {"paper_white_rgb": rgb}
    if not confirm_values(session.console, "White Reference", data):
        return

    cal_dict = session.cal.model_dump(mode="json")
    cal_dict["paper_white_rgb"] = list(rgb)
    session.cal = CalibrationV1(**cal_dict)
    session.modified = True
    session.invalidate_simulator()
    session.save()


# ============================================================================
# Block 2 -- Dot sheet
# ============================================================================

def block_dot_sheet(session: CalibrationSession) -> None:
    """Enter dot measurements, fit radius and profile parameters."""
    session.console.print(Panel(
        "Enter measurements from the printed dot sheet.\n"
        "For each dot: z height, visible diameter, and\n"
        "centre / mid-ring / edge-ring / background RGB.",
        title="Block 2: Dot Sheet",
    ))

    cal_dict = session.cal.model_dump(mode="json")
    dots = dict(cal_dict.get("raw_measurements", {}).get("dots", {}))

    while True:
        action = questionary.select(
            "Dot sheet action:",
            choices=["Add measurement", "Fit & preview", "Back"],
        ).ask()
        if action == "Back":
            break

        if action == "Add measurement":
            test_id = questionary.text(
                "Test ID (e.g. D_z6_01):",
            ).ask()
            if not test_id:
                continue
            z_mm = ask_float("Z height (mm)", 0.0, 30.0)
            cmy = ask_cmy("Color recipe CMY")
            diameter = ask_float("Visible diameter (mm)", 0.1, 50.0)
            center = ask_rgb("Centre RGB")
            mid = ask_rgb("Mid-ring RGB")
            edge = ask_rgb("Edge-ring RGB")
            bg = ask_rgb("Background RGB")

            entry = {
                "test_id": test_id,
                "z_mm": z_mm,
                "color_recipe_cmy": list(cmy),
                "diameter_mm": diameter,
                "center_rgb": list(center),
                "mid_rgb": list(mid),
                "edge_rgb": list(edge),
                "background_rgb": list(bg),
            }
            if not confirm_values(
                session.console, f"Dot {test_id}", entry,
            ):
                continue
            dots[test_id] = entry

            cal_dict["raw_measurements"]["dots"] = dots
            session.cal = CalibrationV1(**cal_dict)
            session.modified = True

        elif action == "Fit & preview":
            if not dots:
                session.console.print("[yellow]No dot measurements yet.[/yellow]")
                continue
            _fit_dots(session, dots, cal_dict)

    if session.modified:
        session.save()


def _fit_dots(
    session: CalibrationSession,
    dots: Dict[str, Any],
    cal_dict: Dict,
) -> None:
    """Fit radius_lut_mm from measured dot diameters."""
    z_grid = cal_dict["z_grid_mm"]

    z_measured = []
    r_measured = []
    for dot in dots.values():
        z_measured.append(dot["z_mm"])
        r_measured.append(dot["diameter_mm"] / 2.0)

    z_arr = np.array(z_measured)
    r_arr = np.array(r_measured)
    sort_idx = np.argsort(z_arr)
    z_arr = z_arr[sort_idx]
    r_arr = r_arr[sort_idx]

    new_radius = np.interp(z_grid, z_arr, r_arr).tolist()

    table = Table(title="Fitted radius_lut_mm")
    table.add_column("z (mm)")
    table.add_column("old R (mm)")
    table.add_column("new R (mm)")
    for i, z in enumerate(z_grid):
        old_r = cal_dict["radius_lut_mm"][i]
        table.add_row(f"{z:.1f}", f"{old_r:.3f}", f"{new_radius[i]:.3f}")
    session.console.print(table)

    # Fit profile shape from centre/mid/edge ratios (simplified)
    _fit_profile_from_dots(session, dots, cal_dict)

    if questionary.confirm("Accept fitted radius?", default=True).ask():
        cal_dict["radius_lut_mm"] = new_radius
        session.cal = CalibrationV1(**cal_dict)
        session.modified = True
        session.invalidate_simulator()


def _fit_profile_from_dots(
    session: CalibrationSession,
    dots: Dict[str, Any],
    cal_dict: Dict,
) -> None:
    """Estimate profile shape from centre/mid/edge darkness ratios.

    Uses a simple grid search over (core_frac, skirt_sigma_frac, skirt_power)
    to minimise the RMS error between measured and predicted radial samples.
    """
    samples = []
    for dot in dots.values():
        bg_lum = _lum(dot["background_rgb"])
        if bg_lum < 0.01:
            continue
        center_rel = 1.0 - _lum(dot["center_rgb"]) / bg_lum
        mid_rel = 1.0 - _lum(dot["mid_rgb"]) / bg_lum
        edge_rel = 1.0 - _lum(dot["edge_rgb"]) / bg_lum
        # normalised u positions: centre=0, mid=0.5, edge=0.9
        samples.append((center_rel, mid_rel, edge_rel))

    if not samples:
        return

    avg = np.mean(samples, axis=0)
    target_center, target_mid, target_edge = avg

    best_err = float("inf")
    best_params: Dict[str, float] = {}

    for cf in np.linspace(0.15, 0.60, 10):
        for sf in np.linspace(0.10, 0.60, 10):
            for sp in np.linspace(1.0, 3.0, 5):
                phi_center = 1.0
                u_mid = 0.5
                u_edge = 0.9
                phi_mid = (
                    1.0 if u_mid <= cf
                    else math.exp(-((u_mid - cf) / max(sf, 1e-6)) ** sp)
                )
                phi_edge = (
                    1.0 if u_edge <= cf
                    else math.exp(-((u_edge - cf) / max(sf, 1e-6)) ** sp)
                )
                if target_center > 0:
                    phi_center_n = phi_center
                    phi_mid_n = phi_mid / phi_center * target_center
                    phi_edge_n = phi_edge / phi_center * target_center
                else:
                    phi_mid_n = phi_mid
                    phi_edge_n = phi_edge
                    phi_center_n = 1.0

                err = (
                    (phi_center_n - target_center) ** 2
                    + (phi_mid_n - target_mid) ** 2
                    + (phi_edge_n - target_edge) ** 2
                )
                if err < best_err:
                    best_err = err
                    best_params = {
                        "core_frac": float(cf),
                        "skirt_sigma_frac": float(sf),
                        "skirt_power": float(sp),
                    }

    if best_params:
        session.console.print(
            f"  Best-fit profile: core={best_params['core_frac']:.2f}  "
            f"sigma={best_params['skirt_sigma_frac']:.2f}  "
            f"power={best_params['skirt_power']:.1f}  "
            f"(RMS err={math.sqrt(best_err):.4f})"
        )
        if questionary.confirm(
            "Apply fitted profile parameters?", default=True,
        ).ask():
            cal_dict["profile"]["core_frac"] = best_params["core_frac"]
            cal_dict["profile"]["skirt_sigma_frac"] = best_params[
                "skirt_sigma_frac"
            ]
            cal_dict["profile"]["skirt_power"] = best_params["skirt_power"]


def _lum(rgb) -> float:
    """Rec.709 luminance from an RGB tuple or list."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# ============================================================================
# Block 3 -- Line sheet
# ============================================================================

def block_line_sheet(session: CalibrationSession) -> None:
    """Enter line measurements, fit mass_lut."""
    session.console.print(Panel(
        "Enter measurements from the printed line sheet.\n"
        "For each line: z, speed, visible width, and\n"
        "centre / shoulder / edge / background RGB\n"
        "(middle section of the line only).",
        title="Block 3: Line Sheet",
    ))

    cal_dict = session.cal.model_dump(mode="json")
    lines = dict(cal_dict.get("raw_measurements", {}).get("lines", {}))

    while True:
        action = questionary.select(
            "Line sheet action:",
            choices=["Add measurement", "Fit & preview", "Back"],
        ).ask()
        if action == "Back":
            break

        if action == "Add measurement":
            test_id = questionary.text("Test ID (e.g. L_z10_v60):").ask()
            if not test_id:
                continue
            z_mm = ask_float("Z height (mm)", 0.0, 30.0)
            speed = ask_float("Speed (mm/s)", 1.0, 300.0)
            cmy = ask_cmy("Color recipe CMY")
            width = ask_float("Visible width (mm)", 0.1, 50.0)
            center = ask_rgb("Centre RGB (mid-section)")
            shoulder = ask_rgb("Shoulder RGB")
            edge = ask_rgb("Edge RGB")
            bg = ask_rgb("Background RGB")

            entry = {
                "test_id": test_id,
                "z_mm": z_mm,
                "speed_mm_s": speed,
                "color_recipe_cmy": list(cmy),
                "width_mm": width,
                "center_rgb": list(center),
                "shoulder_rgb": list(shoulder),
                "edge_rgb": list(edge),
                "background_rgb": list(bg),
            }
            if not confirm_values(
                session.console, f"Line {test_id}", entry,
            ):
                continue
            lines[test_id] = entry

            cal_dict["raw_measurements"]["lines"] = lines
            session.cal = CalibrationV1(**cal_dict)
            session.modified = True

        elif action == "Fit & preview":
            if not lines:
                session.console.print(
                    "[yellow]No line measurements yet.[/yellow]",
                )
                continue
            _fit_lines(session, lines, cal_dict)

    if session.modified:
        session.save()


def _fit_lines(
    session: CalibrationSession,
    lines: Dict[str, Any],
    cal_dict: Dict,
) -> None:
    """Fit mass_lut from measured line centre darkness.

    For each (z, speed) measurement the target centre transmission
    ``T_center = center_lum / bg_lum`` is used to solve for mass:

        T = (1 - alpha) + alpha * paint_lum
        alpha ~ k * mass * ds * phi_center / integral

    We invert this for ``mass`` at the centre (phi_center ~ 1).
    """
    z_grid = np.array(cal_dict["z_grid_mm"])
    speed_grid = np.array(cal_dict["speed_grid_mm_s"])
    old_mass = np.array(cal_dict["mass_lut"])
    new_mass = old_mass.copy()

    k = cal_dict["k_mass"]
    profile_integral = _approx_profile_integral(cal_dict["profile"])
    ds_approx = cal_dict["sampling"]["max_step_mm"]

    # measured points
    z_pts = []
    v_pts = []
    mass_pts = []

    for line in lines.values():
        bg_lum = _lum(line["background_rgb"])
        center_lum = _lum(line["center_rgb"])
        if bg_lum < 0.01:
            continue
        T_center = center_lum / bg_lum
        T_center = max(T_center, 0.001)

        paint_lum = 0.0  # worst case: black paint
        # T = (1-a) + a*paint  =>  a = (1 - T) / (1 - paint)
        alpha_target = (1.0 - T_center) / max(1.0 - paint_lum, 1e-6)
        alpha_target = min(max(alpha_target, 0.0), 1.0)

        if profile_integral > 0 and k > 0 and ds_approx > 0:
            mass_val = alpha_target * profile_integral / (k * ds_approx)
        else:
            mass_val = alpha_target

        z_pts.append(line["z_mm"])
        v_pts.append(line["speed_mm_s"])
        mass_pts.append(mass_val)

    if not z_pts:
        session.console.print("[yellow]Could not derive mass values.[/yellow]")
        return

    # scatter-interpolate onto the grid
    for iz, zg in enumerate(z_grid):
        for iv, vg in enumerate(speed_grid):
            weights = []
            values = []
            for zp, vp, mp in zip(z_pts, v_pts, mass_pts):
                dz = abs(zp - zg) / max(z_grid[-1] - z_grid[0], 1.0)
                dv = abs(vp - vg) / max(
                    speed_grid[-1] - speed_grid[0], 1.0,
                )
                dist = math.sqrt(dz * dz + dv * dv) + 1e-6
                w = 1.0 / (dist * dist)
                weights.append(w)
                values.append(mp)
            total_w = sum(weights)
            new_mass[iz, iv] = sum(
                w * v for w, v in zip(weights, values)
            ) / total_w

    table = Table(title="Fitted mass_lut")
    table.add_column("z \\ v", style="cyan")
    for vg in speed_grid:
        table.add_column(f"{vg:.0f}")
    for iz, zg in enumerate(z_grid):
        row = [f"{zg:.1f}"]
        for iv in range(len(speed_grid)):
            row.append(f"{new_mass[iz, iv]:.4f}")
        table.add_row(*row)
    session.console.print(table)

    if questionary.confirm("Accept fitted mass_lut?", default=True).ask():
        cal_dict["mass_lut"] = new_mass.tolist()
        session.cal = CalibrationV1(**cal_dict)
        session.modified = True
        session.invalidate_simulator()


def _approx_profile_integral(profile_dict: Dict) -> float:
    """Quick numerical integral of the radial profile."""
    N = 512
    margin = profile_dict.get("margin_factor", 1.5)
    core = profile_dict.get("core_frac", 0.4)
    sigma = max(profile_dict.get("skirt_sigma_frac", 0.28), 1e-8)
    power = profile_dict.get("skirt_power", 1.8)

    u = np.linspace(0, margin, N)
    phi = np.ones(N)
    skirt = u > core
    phi[skirt] = np.exp(-((u[skirt] - core) / sigma) ** power)
    phi[u > margin] = 0.0

    du = u[1] - u[0]
    return float(2.0 * np.pi * np.trapz(phi * u, dx=du))


# ============================================================================
# Block 4 -- Color swatches
# ============================================================================

def block_color_swatches(session: CalibrationSession) -> None:
    """Enter swatch measurements, fit color LUT."""
    session.console.print(Panel(
        "Enter measurements from the printed color swatch chart.\n"
        "For each swatch: CMY recipe and measured interior RGB.",
        title="Block 4: Color Swatches",
    ))

    cal_dict = session.cal.model_dump(mode="json")
    swatches = dict(
        cal_dict.get("raw_measurements", {}).get("swatches", {}),
    )

    while True:
        action = questionary.select(
            "Swatch action:",
            choices=["Add measurement", "Fit & preview", "Back"],
        ).ask()
        if action == "Back":
            break

        if action == "Add measurement":
            recipe_id = questionary.text(
                "Recipe ID (e.g. S_C100_M000_Y000):",
            ).ask()
            if not recipe_id:
                continue
            cmy = ask_cmy("CMY command")
            rgb = ask_rgb("Interior RGB (measured)")

            entry = {
                "recipe_id": recipe_id,
                "cmy_command": list(cmy),
                "interior_rgb": list(rgb),
            }
            if not confirm_values(
                session.console, f"Swatch {recipe_id}", entry,
            ):
                continue
            swatches[recipe_id] = entry

            cal_dict["raw_measurements"]["swatches"] = swatches
            session.cal = CalibrationV1(**cal_dict)
            session.modified = True

        elif action == "Fit & preview":
            if not swatches:
                session.console.print(
                    "[yellow]No swatch measurements yet.[/yellow]",
                )
                continue
            _fit_swatches(session, swatches, cal_dict)

    if session.modified:
        session.save()


def _fit_swatches(
    session: CalibrationSession,
    swatches: Dict[str, Any],
    cal_dict: Dict,
) -> None:
    """Build a color LUT from measured swatches.

    For each swatch at grid point (c, m, y), set the LUT entry directly.
    Between measurements, use inverse-distance interpolation.
    """
    ca = cal_dict["color_axes"]
    Nc, Nm, Ny = ca["c_steps"], ca["m_steps"], ca["y_steps"]
    lo, hi = ca["domain"]

    lut = np.zeros((Nc, Nm, Ny, 3), dtype=np.float32)
    # default: subtractive
    c_vals = np.linspace(lo, hi, Nc)
    m_vals = np.linspace(lo, hi, Nm)
    y_vals = np.linspace(lo, hi, Ny)
    C, M, Y = np.meshgrid(c_vals, m_vals, y_vals, indexing="ij")
    lut[..., 0] = 1.0 - C
    lut[..., 1] = 1.0 - M
    lut[..., 2] = 1.0 - Y

    measured_points = []
    for sw in swatches.values():
        cmy = sw["cmy_command"]
        rgb = sw["interior_rgb"]
        measured_points.append((cmy, rgb))

    if measured_points:
        for ic in range(Nc):
            for im in range(Nm):
                for iy in range(Ny):
                    gc = c_vals[ic]
                    gm = m_vals[im]
                    gy = y_vals[iy]
                    weights = []
                    values = []
                    for (mc, mr) in measured_points:
                        dc = gc - mc[0]
                        dm = gm - mc[1]
                        dy = gy - mc[2]
                        dist = math.sqrt(dc*dc + dm*dm + dy*dy) + 1e-6
                        w = 1.0 / (dist * dist)
                        weights.append(w)
                        values.append(mr)
                    tw = sum(weights)
                    for ch in range(3):
                        lut[ic, im, iy, ch] = sum(
                            w * v[ch] for w, v in zip(weights, values)
                        ) / tw

    lut = np.clip(lut, 0.0, 1.0)

    # Save as .pt
    lut_path = session.cal_path.parent / "color_lut.pt"
    torch.save(torch.from_numpy(lut), str(lut_path))

    cal_dict["color_lut_path"] = str(lut_path)
    session.cal = CalibrationV1(**cal_dict)
    session.modified = True
    session.invalidate_simulator()

    # preview table
    table = Table(title="Swatch comparison")
    table.add_column("Recipe")
    table.add_column("Target RGB")
    table.add_column("LUT RGB")
    table.add_column("dE", justify="right")

    for sid, sw in swatches.items():
        cmy = sw["cmy_command"]
        target = sw["interior_rgb"]
        # nearest LUT lookup
        ic = int(round((cmy[0] - lo) / (hi - lo) * (Nc - 1)))
        im = int(round((cmy[1] - lo) / (hi - lo) * (Nm - 1)))
        iy = int(round((cmy[2] - lo) / (hi - lo) * (Ny - 1)))
        ic = max(0, min(ic, Nc - 1))
        im = max(0, min(im, Nm - 1))
        iy = max(0, min(iy, Ny - 1))
        lut_rgb = lut[ic, im, iy].tolist()

        t_t = torch.tensor([target], dtype=torch.float32)
        l_t = torch.tensor([lut_rgb], dtype=torch.float32)
        lab_t = color_utils.rgb_to_lab(t_t)
        lab_l = color_utils.rgb_to_lab(l_t)
        de = float(color_utils.delta_e2000(lab_t, lab_l).item())

        table.add_row(
            sid,
            f"({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})",
            f"({lut_rgb[0]:.2f},{lut_rgb[1]:.2f},{lut_rgb[2]:.2f})",
            f"{de:.2f}",
        )
    session.console.print(table)
    session.console.print(f"[green]Color LUT saved to {lut_path}[/green]")


# ============================================================================
# Block 5 -- Layering
# ============================================================================

def block_layering(session: CalibrationSession) -> None:
    """Enter overlap measurements, validate layering model."""
    session.console.print(Panel(
        "Enter measurements from the layering staircase pattern.\n"
        "For each overlap region: colours involved, single-colour\n"
        "RGB, and overlap-region RGB.",
        title="Block 5: Layering",
    ))

    cal_dict = session.cal.model_dump(mode="json")
    overlaps = dict(
        cal_dict.get("raw_measurements", {}).get("overlaps", {}),
    )

    while True:
        action = questionary.select(
            "Layering action:",
            choices=["Add measurement", "Validate", "Back"],
        ).ask()
        if action == "Back":
            break

        if action == "Add measurement":
            pattern_id = questionary.text(
                "Pattern ID (e.g. O_CM_01):",
            ).ask()
            if not pattern_id:
                continue
            n_colors = int(ask_float("Number of colours (2-4)", 2, 4))
            colors = []
            single_rgbs = []
            for ci in range(n_colors):
                c = ask_cmy(f"  Colour {ci+1} CMY")
                colors.append(list(c))
                rgb = ask_rgb(f"  Single-colour RGB for colour {ci+1}")
                single_rgbs.append(list(rgb))

            n_overlaps = int(
                ask_float("Number of overlap regions", 1, 10),
            )
            overlap_rgbs = []
            for oi in range(n_overlaps):
                rgb = ask_rgb(f"  Overlap region {oi+1} RGB")
                overlap_rgbs.append(list(rgb))

            entry = {
                "pattern_id": pattern_id,
                "colors": colors,
                "single_rgbs": single_rgbs,
                "overlap_rgbs": overlap_rgbs,
            }
            if not confirm_values(
                session.console, f"Overlap {pattern_id}", entry,
            ):
                continue
            overlaps[pattern_id] = entry
            cal_dict["raw_measurements"]["overlaps"] = overlaps
            session.cal = CalibrationV1(**cal_dict)
            session.modified = True

        elif action == "Validate":
            if not overlaps:
                session.console.print(
                    "[yellow]No overlap measurements yet.[/yellow]",
                )
                continue
            _validate_layering(session, overlaps, cal_dict)

    if session.modified:
        session.save()


def _validate_layering(
    session: CalibrationSession,
    overlaps: Dict[str, Any],
    cal_dict: Dict,
) -> None:
    """Compare simulated overlap RGB to measured, offer gain correction."""
    sim = session.simulator
    delta_es = []

    table = Table(title="Layering Validation")
    table.add_column("Pattern")
    table.add_column("Region")
    table.add_column("Measured RGB")
    table.add_column("Simulated RGB")
    table.add_column("dE", justify="right")

    for pid, ov in overlaps.items():
        colors = ov["colors"]
        # simulate overlap by sequential rendering
        canvas = sim.reset(batch_size=1)
        fill_z = 6.0
        fill_speed = 30.0
        cx = sim._cal.render.work_area_mm[0] / 2
        cy = sim._cal.render.work_area_mm[1] / 2
        line_len = 20.0

        for cidx, cmy in enumerate(colors):
            stroke = {
                "bezier": {
                    "p1": (cx - line_len / 2, cy),
                    "p2": (cx - line_len / 6, cy),
                    "p3": (cx + line_len / 6, cy),
                    "p4": (cx + line_len / 2, cy),
                },
                "z_profile": {"z0": fill_z, "z1": fill_z},
                "speed_profile": {"v0": fill_speed, "v1": fill_speed},
                "color_cmy": {"c": cmy[0], "m": cmy[1], "y": cmy[2]},
            }
            # render several parallel passes to build coverage
            for offset in np.arange(-3.0, 3.0, 0.5):
                s = copy.deepcopy(stroke)
                for key in ["p1", "p2", "p3", "p4"]:
                    x, y = s["bezier"][key]
                    s["bezier"][key] = (x, y + offset)
                canvas = sim.render_stroke(canvas, s)

        # sample overlap centre
        px_x = int(cx * sim._dpi_x)
        px_y = int(cy * sim._dpi_y)
        px_x = min(max(px_x, 0), sim._W - 1)
        px_y = min(max(px_y, 0), sim._H - 1)
        sim_rgb = canvas[0, :, px_y, px_x].cpu().tolist()

        for oi, meas_rgb in enumerate(ov["overlap_rgbs"]):
            t = torch.tensor([meas_rgb], dtype=torch.float32)
            s = torch.tensor([sim_rgb], dtype=torch.float32)
            lab_t = color_utils.rgb_to_lab(t)
            lab_s = color_utils.rgb_to_lab(s)
            de = float(color_utils.delta_e2000(lab_t, lab_s).item())
            delta_es.append(de)

            table.add_row(
                pid, f"overlap_{oi}",
                f"({meas_rgb[0]:.2f},{meas_rgb[1]:.2f},{meas_rgb[2]:.2f})",
                f"({sim_rgb[0]:.2f},{sim_rgb[1]:.2f},{sim_rgb[2]:.2f})",
                f"{de:.2f}",
            )

    session.console.print(table)

    if delta_es:
        mean_de = sum(delta_es) / len(delta_es)
        session.console.print(
            f"Mean dE across overlaps: {mean_de:.2f}",
        )
        if mean_de > 3.0:
            session.console.print(
                "[yellow]Systematic bias detected. "
                "Consider fitting layer_gain_lut.[/yellow]",
            )
            if questionary.confirm(
                "Fit a simple layer_gain_lut?", default=False,
            ).ask():
                gain = 1.0 / max(mean_de / 3.0, 0.5)
                cal_dict["layer_gain_lut"] = [1.0, gain, gain * 0.9, gain * 0.8]
                session.cal = CalibrationV1(**cal_dict)
                session.modified = True
                session.invalidate_simulator()


# ============================================================================
# Block 6 -- Preview / validate
# ============================================================================

def block_preview_validate(session: CalibrationSession) -> None:
    """Rebuild simulator and compare all measurements."""
    session.console.print(Panel(
        "Rebuilding simulator from current calibration\n"
        "and comparing against all raw measurements.",
        title="Block 6: Preview / Validate",
    ))

    session.invalidate_simulator()
    if session.modified:
        session.save()

    sim = session.simulator
    results = sim.compare_to_measurements()
    table = sim.summary_table()
    session.console.print(table)

    # aggregate stats
    all_de = []
    for block_data in results.values():
        for entry in block_data.values():
            all_de.append(entry["delta_e"])

    cal_dict = session.cal.model_dump(mode="json")

    if all_de:
        mean_de = sum(all_de) / len(all_de)
        max_de = max(all_de)
        session.console.print(
            f"\nOverall  mean dE = {mean_de:.2f}   max dE = {max_de:.2f}",
        )
        cal_dict["validation"]["last_run"] = (
            datetime.datetime.now().isoformat()
        )
        cal_dict["validation"]["overall_delta_e_mean"] = round(mean_de, 3)
        cal_dict["validation"]["overall_delta_e_max"] = round(max_de, 3)
    else:
        session.console.print(
            "[yellow]No measurements to validate against.[/yellow]",
        )
        cal_dict["validation"]["last_run"] = (
            datetime.datetime.now().isoformat()
        )

    session.cal = CalibrationV1(**cal_dict)
    session.modified = True

    # save preview images
    out_dir = Path(session.cal.preview_settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        z_vals = session.cal.z_grid_mm[:5]
        dot_img = sim.render_dot_sheet(z_vals)
        _save_canvas_png(dot_img, out_dir / "dot_sheet.png")

        speed_vals = session.cal.speed_grid_mm_s[:3]
        line_img = sim.render_line_sheet(z_vals[:3], speed_vals)
        _save_canvas_png(line_img, out_dir / "line_sheet.png")

        session.console.print(
            f"[green]Preview images saved to {out_dir}/[/green]",
        )
    except Exception as exc:
        session.console.print(
            f"[red]Preview render failed: {exc}[/red]",
        )

    session.save()


def _save_canvas_png(
    canvas: torch.Tensor, path: Path,
) -> None:
    """Save a (1,3,H,W) linear-RGB canvas as an sRGB PNG."""
    from PIL import Image

    img = canvas[0].clamp(0.0, 1.0).cpu()
    img_srgb = color_utils.linear_to_srgb(img.permute(1, 2, 0))
    arr = (img_srgb.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


# ============================================================================
# Calibration pattern definitions (data-only)
# ============================================================================

def dot_matrix_pattern(
    z_values: List[float],
    color_cmy: Tuple[float, float, float] = (0.8, 0.2, 0.1),
    spacing_mm: float = 20.0,
    n_repeats: int = 3,
    speed: float = 60.0,
) -> List[Dict]:
    """Generate stroke descriptors for a dot matrix calibration sheet.

    Returns
    -------
    list of dict
        Stroke descriptors (tiny-length strokes -> isolated dots).
    """
    strokes = []
    c, m, y = color_cmy
    for row, z in enumerate(z_values):
        cy = spacing_mm * (row + 1)
        for col in range(n_repeats):
            cx = spacing_mm * (col + 1)
            tiny = 0.01
            strokes.append({
                "bezier": {
                    "p1": (cx, cy), "p2": (cx + tiny, cy),
                    "p3": (cx + tiny, cy), "p4": (cx + tiny, cy),
                },
                "z_profile": {"z0": z, "z1": z},
                "speed_profile": {"v0": speed, "v1": speed},
                "color_cmy": {"c": c, "m": m, "y": y},
            })
    return strokes


def line_matrix_pattern(
    z_values: List[float],
    speed_values: List[float],
    color_cmy: Tuple[float, float, float] = (0.8, 0.2, 0.1),
    length_mm: float = 40.0,
    spacing_mm: float = 20.0,
) -> List[Dict]:
    """Generate stroke descriptors for a line matrix calibration sheet."""
    strokes = []
    c, m, y = color_cmy
    margin = spacing_mm
    for ri, z in enumerate(z_values):
        cy = margin + ri * spacing_mm
        for ci, spd in enumerate(speed_values):
            cx0 = margin + ci * (length_mm + margin)
            cx1 = cx0 + length_mm
            strokes.append({
                "bezier": {
                    "p1": (cx0, cy),
                    "p2": (cx0 + length_mm / 3, cy),
                    "p3": (cx0 + 2 * length_mm / 3, cy),
                    "p4": (cx1, cy),
                },
                "z_profile": {"z0": z, "z1": z},
                "speed_profile": {"v0": spd, "v1": spd},
                "color_cmy": {"c": c, "m": m, "y": y},
            })
    return strokes


def color_swatch_pattern(
    cmy_recipes: List[Tuple[float, float, float]],
    swatch_size_mm: float = 15.0,
    fill_speed: float = 30.0,
    fill_z: float = 6.0,
    line_spacing_mm: float = 0.5,
) -> List[Dict]:
    """Generate dense-fill strokes for color swatches."""
    strokes = []
    margin = 10.0
    cols = max(1, int(200.0 / (swatch_size_mm + margin)))
    for idx, (c, m, y) in enumerate(cmy_recipes):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (swatch_size_mm + margin)
        y0 = margin + row * (swatch_size_mm + margin)
        yy = y0
        while yy < y0 + swatch_size_mm:
            strokes.append({
                "bezier": {
                    "p1": (x0, yy),
                    "p2": (x0 + swatch_size_mm / 3, yy),
                    "p3": (x0 + 2 * swatch_size_mm / 3, yy),
                    "p4": (x0 + swatch_size_mm, yy),
                },
                "z_profile": {"z0": fill_z, "z1": fill_z},
                "speed_profile": {"v0": fill_speed, "v1": fill_speed},
                "color_cmy": {"c": c, "m": m, "y": y},
            })
            yy += line_spacing_mm
    return strokes


def layering_staircase_pattern(
    colors: List[Tuple[float, float, float]],
    rect_size_mm: float = 30.0,
    overlap_mm: float = 15.0,
    fill_speed: float = 30.0,
    fill_z: float = 6.0,
    line_spacing_mm: float = 0.5,
) -> List[Dict]:
    """Generate nested-rectangle strokes for layering validation."""
    strokes = []
    margin = 10.0
    for cidx, (c, m, y) in enumerate(colors):
        x0 = margin + cidx * overlap_mm
        y0 = margin
        yy = y0
        while yy < y0 + rect_size_mm:
            strokes.append({
                "bezier": {
                    "p1": (x0, yy),
                    "p2": (x0 + rect_size_mm / 3, yy),
                    "p3": (x0 + 2 * rect_size_mm / 3, yy),
                    "p4": (x0 + rect_size_mm, yy),
                },
                "z_profile": {"z0": fill_z, "z1": fill_z},
                "speed_profile": {"v0": fill_speed, "v1": fill_speed},
                "color_cmy": {"c": c, "m": m, "y": y},
            })
            yy += line_spacing_mm
    return strokes


# ============================================================================
# Main menu
# ============================================================================

def main(cal_path: Optional[str] = None) -> None:
    """Launch the calibration wizard."""
    console = Console()
    console.clear()
    console.print(Panel(
        "[bold]Digital Twin Calibration Wizard[/bold]\n"
        "GPU stamp-splatting airbrush simulator",
        style="cyan",
    ))

    if cal_path is None:
        default = str(DEFAULT_CALIBRATION_PATH)
        cal_path = questionary.path(
            "Calibration YAML path:", default=default,
        ).ask()
    if not cal_path:
        return

    cal_path_obj = Path(cal_path)
    if not cal_path_obj.exists():
        console.print(f"[red]File not found: {cal_path}[/red]")
        return

    session = CalibrationSession(cal_path_obj, console)

    menu_items = [
        "1. White reference",
        "2. Dot sheet",
        "3. Line sheet",
        "4. Color swatches",
        "5. Layering",
        "6. Preview / validate",
        "Quit",
    ]

    handlers = {
        "1. White reference": block_white_reference,
        "2. Dot sheet": block_dot_sheet,
        "3. Line sheet": block_line_sheet,
        "4. Color swatches": block_color_swatches,
        "5. Layering": block_layering,
        "6. Preview / validate": block_preview_validate,
    }

    while True:
        console.print()
        choice = questionary.select(
            "Calibration block:", choices=menu_items,
        ).ask()
        if choice is None or choice == "Quit":
            break
        handler = handlers.get(choice)
        if handler:
            try:
                handler(session)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                logger.exception("Block failed")

    if session.modified:
        if questionary.confirm("Save unsaved changes?", default=True).ask():
            session.save()

    console.print("[bold]Done.[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Digital Twin Calibration Wizard",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to calibration YAML",
    )
    args = parser.parse_args()
    main(cal_path=args.config)
