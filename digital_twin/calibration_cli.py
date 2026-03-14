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
import datetime
import logging
import math
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


def _format_value(v: Any) -> str:
    """Format a value for display in confirm_values, handling nested lists."""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (tuple, list)):
        if v and isinstance(v[0], (tuple, list)):
            # nested: list of lists/tuples
            parts = []
            for sub in v:
                inner = ", ".join(_format_value(x) for x in sub)
                parts.append(f"[{inner}]")
            return ", ".join(parts)
        return ", ".join(_format_value(x) for x in v)
    return str(v)


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
        table.add_row(k, _format_value(v))
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
                session.console.print(
                    "[yellow]No dot measurements yet.[/yellow]",
                )
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

    Uses a grid search over (core_frac, skirt_sigma_frac, skirt_power)
    to minimise the RMS error between measured and predicted radial
    samples normalised by the measured background.
    """
    samples = []
    for dot in dots.values():
        bg_lum = _lum(dot["background_rgb"])
        if bg_lum < 0.01:
            continue
        center_rel = 1.0 - _lum(dot["center_rgb"]) / bg_lum
        mid_rel = 1.0 - _lum(dot["mid_rgb"]) / bg_lum
        edge_rel = 1.0 - _lum(dot["edge_rgb"]) / bg_lum
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
                u_mid, u_edge = 0.5, 0.9
                phi_mid = (
                    1.0 if u_mid <= cf
                    else math.exp(
                        -((u_mid - cf) / max(sf, 1e-6)) ** sp
                    )
                )
                phi_edge = (
                    1.0 if u_edge <= cf
                    else math.exp(
                        -((u_edge - cf) / max(sf, 1e-6)) ** sp
                    )
                )
                if target_center > 0:
                    phi_mid_n = phi_mid / phi_center * target_center
                    phi_edge_n = phi_edge / phi_center * target_center
                else:
                    phi_mid_n = phi_mid
                    phi_edge_n = phi_edge

                err = (
                    (phi_mid_n - target_mid) ** 2
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

    For each (z, speed) measurement, the centre transmission
    ``T = center_lum / bg_lum`` is inverted using the correct
    paint luminance (from subtractive model of the line's CMY
    recipe) and the renderer's adaptive ds (based on radius at z).

    The inversion accounts for the R² area normalisation and the
    number of overlapping stamps at the line centre.
    """
    z_grid = np.array(cal_dict["z_grid_mm"])
    speed_grid = np.array(cal_dict["speed_grid_mm_s"])
    old_mass = np.array(cal_dict["mass_lut"])
    new_mass = old_mass.copy()

    k = cal_dict["k_mass"]
    radius_lut = np.array(cal_dict["radius_lut_mm"])
    profile_integral = _approx_profile_integral(cal_dict["profile"])
    max_step = cal_dict["sampling"]["max_step_mm"]
    margin = cal_dict["profile"].get("margin_factor", 1.5)

    z_pts: List[float] = []
    v_pts: List[float] = []
    mass_pts: List[float] = []

    for line in lines.values():
        bg_lum = _lum(line["background_rgb"])
        center_lum = _lum(line["center_rgb"])
        if bg_lum < 0.01:
            continue

        T_center = max(center_lum / bg_lum, 0.001)

        # actual paint luminance from subtractive model
        cmy = line["color_recipe_cmy"]
        paint_lum = (
            0.2126 * (1.0 - cmy[0])
            + 0.7152 * (1.0 - cmy[1])
            + 0.0722 * (1.0 - cmy[2])
        )

        # adaptive ds matching the renderer's rule
        z = line["z_mm"]
        R = float(np.interp(z, z_grid, radius_lut))
        R = max(R, 0.01)
        target_ds = min(0.25 * R, max_step)
        actual_ds = max(target_ds, 1e-4)

        R_sq = R * R

        # estimate how many stamps overlap the centre pixel
        n_covering = max(1, int(2.0 * margin * R / actual_ds))

        # small-alpha approximation:
        #   T ≈ 1 - n * alpha_single * (1 - paint_lum)
        #   alpha_single = k * mass * ds / (integral * R²)
        one_minus_T = max(1.0 - T_center, 0.0)
        denom = max(
            n_covering * k * actual_ds * (1.0 - paint_lum), 1e-8,
        )
        mass_val = one_minus_T * profile_integral * R_sq / denom

        z_pts.append(z)
        v_pts.append(line["speed_mm_s"])
        mass_pts.append(mass_val)

    if not z_pts:
        session.console.print(
            "[yellow]Could not derive mass values.[/yellow]",
        )
        return

    # inverse-distance interpolation onto the (z, speed) grid
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
        "For each swatch: CMY recipe and measured interior RGB.\n\n"
        "The fitter inverts the compositing model to recover\n"
        "per-stamp paint_rgb from the observed dense-fill colour.",
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

    The measured interior RGB is the result of accumulated compositing
    over many fill strokes, not the per-stamp paint_rgb.  This function
    estimates the per-stamp alpha from the current mass_lut and fill
    conditions, then inverts the compositing to recover paint_rgb.

    At grid points without nearby measurements the default subtractive
    model (RGB = 1 - CMY) is used as a fallback.
    """
    ca = cal_dict["color_axes"]
    Nc, Nm, Ny = ca["c_steps"], ca["m_steps"], ca["y_steps"]
    lo, hi = ca["domain"]

    # default LUT: subtractive
    c_vals = np.linspace(lo, hi, Nc)
    m_vals = np.linspace(lo, hi, Nm)
    y_vals = np.linspace(lo, hi, Ny)
    C, M, Y = np.meshgrid(c_vals, m_vals, y_vals, indexing="ij")
    lut = np.stack([1.0 - C, 1.0 - M, 1.0 - Y], axis=-1).astype(np.float32)

    # estimate per-fill-pass alpha for inversion
    pw = np.array(cal_dict["paper_white_rgb"])
    alpha_eff, n_passes = _estimate_fill_alpha(cal_dict)

    measured_points: List[Tuple[List[float], np.ndarray]] = []
    for sw in swatches.values():
        cmy = sw["cmy_command"]
        interior = np.array(sw["interior_rgb"])
        paint_rgb = _invert_swatch_color(interior, pw, alpha_eff, n_passes)
        measured_points.append((cmy, paint_rgb))

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

    # save as .pt -- store just the filename, not the full relative path
    lut_filename = "color_lut.pt"
    lut_disk_path = session.cal_path.parent / lut_filename
    torch.save(torch.from_numpy(lut), str(lut_disk_path))

    cal_dict["color_lut_path"] = lut_filename
    session.cal = CalibrationV1(**cal_dict)
    session.modified = True
    session.invalidate_simulator()

    # preview table
    table = Table(title="Swatch comparison (per-stamp paint_rgb vs target)")
    table.add_column("Recipe")
    table.add_column("Target interior")
    table.add_column("Inverted paint_rgb")
    table.add_column("dE (interior vs resim)")

    for sid, sw in swatches.items():
        cmy = sw["cmy_command"]
        target = sw["interior_rgb"]
        ic = int(round((cmy[0] - lo) / max(hi - lo, 1e-6) * (Nc - 1)))
        im = int(round((cmy[1] - lo) / max(hi - lo, 1e-6) * (Nm - 1)))
        iy = int(round((cmy[2] - lo) / max(hi - lo, 1e-6) * (Ny - 1)))
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
    session.console.print(
        f"[green]Color LUT saved to {lut_disk_path}[/green]",
    )


def _estimate_fill_alpha(cal_dict: Dict) -> Tuple[float, int]:
    """Estimate per-stamp centre alpha and pass count for a dense fill.

    Uses the preview defaults (z, speed) and the mass_lut to compute
    the alpha a single stamp deposits at the centre pixel, and how
    many stamps overlap a pixel in a dense horizontal fill.
    """
    ps = cal_dict.get("preview_settings", {})
    fill_z = ps.get("default_z_mm", 6.0)
    fill_speed = ps.get("default_speed_mm_s", 30.0)
    line_spacing = 0.5

    z_grid = np.array(cal_dict["z_grid_mm"])
    speed_grid = np.array(cal_dict["speed_grid_mm_s"])
    radius_lut = np.array(cal_dict["radius_lut_mm"])
    mass_arr = np.array(cal_dict["mass_lut"])
    k = cal_dict["k_mass"]

    R = float(np.interp(fill_z, z_grid, radius_lut))
    R = max(R, 0.01)
    margin = cal_dict["profile"].get("margin_factor", 1.5)
    max_step = cal_dict["sampling"]["max_step_mm"]
    target_ds = min(0.25 * R, max_step)
    ds = max(target_ds, 1e-4)

    # bilinear interp of mass
    iz = np.searchsorted(z_grid, fill_z).clip(1, len(z_grid) - 1)
    iv = np.searchsorted(speed_grid, fill_speed).clip(
        1, len(speed_grid) - 1,
    )
    tz = (fill_z - z_grid[iz - 1]) / max(
        z_grid[iz] - z_grid[iz - 1], 1e-6,
    )
    tv = (fill_speed - speed_grid[iv - 1]) / max(
        speed_grid[iv] - speed_grid[iv - 1], 1e-6,
    )
    tz = np.clip(tz, 0, 1)
    tv = np.clip(tv, 0, 1)
    mass = (
        mass_arr[iz-1, iv-1] * (1-tz) * (1-tv)
        + mass_arr[iz-1, iv] * (1-tz) * tv
        + mass_arr[iz, iv-1] * tz * (1-tv)
        + mass_arr[iz, iv] * tz * tv
    )

    integral = _approx_profile_integral(cal_dict["profile"])
    R_sq = R * R
    alpha_single = k * mass * ds / max(integral * R_sq, 1e-8)
    alpha_single = min(alpha_single, 1.0)

    # stamps along the line overlapping a centre pixel
    n_along_line = max(1, int(2 * margin * R / ds))
    # fill lines crossing a pixel (vertical coverage)
    n_lines = max(1, int(2 * margin * R / line_spacing))
    n_passes = n_along_line * n_lines

    return float(alpha_single), n_passes


def _invert_swatch_color(
    interior_rgb: np.ndarray,
    paper_white: np.ndarray,
    alpha_per_stamp: float,
    n_passes: int,
) -> np.ndarray:
    """Invert the compositing model to recover per-stamp paint_rgb.

    The dense swatch is modelled as ``n_passes`` applications of:
        canvas = canvas * ((1-a) + a * paint_rgb)

    After N passes:
        interior = paper * T^N  where  T = (1-a) + a*paint

    We invert for paint_rgb per channel.
    """
    ratio = interior_rgb / np.maximum(paper_white, 1e-6)
    ratio = np.clip(ratio, 0.01, 0.99)

    N = max(n_passes, 1)
    T_eff = np.power(ratio, 1.0 / N)

    a = np.clip(alpha_per_stamp, 0.01, 0.99)
    paint = (T_eff - (1.0 - a)) / a
    return np.clip(paint, 0.0, 1.0)


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
                rgb = ask_rgb(
                    f"  Single-colour RGB for colour {ci+1}",
                )
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
    """Compare simulated overlap RGB to measured for each pattern.

    Each colour is rendered as a dense fill band centred on the canvas.
    Single-colour simulations are compared to the entered single_rgbs.
    The combined overlap is compared to each entered overlap_rgb.
    """
    sim = session.simulator
    delta_es: List[float] = []

    table = Table(title="Layering Validation")
    table.add_column("Pattern")
    table.add_column("Region")
    table.add_column("Measured RGB")
    table.add_column("Simulated RGB")
    table.add_column("dE", justify="right")

    for pid, ov in overlaps.items():
        colors = ov["colors"]
        single_rgbs_meas = ov.get("single_rgbs", [])

        # --- single-colour validation ---
        for ci, (cmy, meas_single) in enumerate(
            zip(colors, single_rgbs_meas),
        ):
            single_canvas = sim.reset(batch_size=1)
            single_canvas = _render_fill_band(
                sim, single_canvas, cmy, band_half=10.0,
            )
            sim_single = _sample_centre(
                sim, single_canvas,
            ).cpu().tolist()

            t = torch.tensor([meas_single], dtype=torch.float32)
            s = torch.tensor([sim_single], dtype=torch.float32)
            lab_t = color_utils.rgb_to_lab(t)
            lab_s = color_utils.rgb_to_lab(s)
            de = float(color_utils.delta_e2000(lab_t, lab_s).item())
            delta_es.append(de)

            table.add_row(
                pid, f"single_c{ci}",
                _fmt_rgb(meas_single), _fmt_rgb(sim_single),
                f"{de:.2f}",
            )

        # --- combined overlap ---
        overlap_canvas = sim.reset(batch_size=1)
        for cmy in colors:
            overlap_canvas = _render_fill_band(
                sim, overlap_canvas, cmy, band_half=10.0,
            )
        sim_overlap = _sample_centre(
            sim, overlap_canvas,
        ).cpu().tolist()

        for oi, meas_rgb in enumerate(ov["overlap_rgbs"]):
            t = torch.tensor([meas_rgb], dtype=torch.float32)
            s = torch.tensor([sim_overlap], dtype=torch.float32)
            lab_t = color_utils.rgb_to_lab(t)
            lab_s = color_utils.rgb_to_lab(s)
            de = float(color_utils.delta_e2000(lab_t, lab_s).item())
            delta_es.append(de)

            table.add_row(
                pid, f"overlap_{oi}",
                _fmt_rgb(meas_rgb), _fmt_rgb(sim_overlap),
                f"{de:.2f}",
            )

    session.console.print(table)

    if delta_es:
        mean_de = sum(delta_es) / len(delta_es)
        session.console.print(
            f"Mean dE across layering: {mean_de:.2f}",
        )
        de_fail = cal_dict.get("preview_settings", {}).get(
            "delta_e_fail_threshold", 5.0,
        )
        if mean_de > de_fail:
            session.console.print(
                "[yellow]Systematic bias detected. "
                "Consider fitting layer_gain_lut.[/yellow]",
            )
            if questionary.confirm(
                "Fit a simple layer_gain_lut?", default=False,
            ).ask():
                gain = 1.0 / max(mean_de / 3.0, 0.5)
                cal_dict["layer_gain_lut"] = [
                    1.0, gain, gain * 0.9, gain * 0.8,
                ]
                session.cal = CalibrationV1(**cal_dict)
                session.modified = True
                session.invalidate_simulator()


def _render_fill_band(
    sim, canvas: torch.Tensor, cmy: list, band_half: float = 10.0,
) -> torch.Tensor:
    """Render a dense horizontal fill band centred on the canvas."""
    ps = sim._cal.preview_settings
    fill_z = ps.default_z_mm
    fill_speed = ps.default_speed_mm_s
    cx = sim._cal.render.work_area_mm[0] / 2
    cy = sim._cal.render.work_area_mm[1] / 2
    line_len = 30.0
    line_spacing = 0.5

    yy = cy - band_half
    while yy < cy + band_half:
        stroke = {
            "bezier": {
                "p1": (cx - line_len / 2, yy),
                "p2": (cx - line_len / 6, yy),
                "p3": (cx + line_len / 6, yy),
                "p4": (cx + line_len / 2, yy),
            },
            "z_profile": {"z0": fill_z, "z1": fill_z},
            "speed_profile": {"v0": fill_speed, "v1": fill_speed},
            "color_cmy": {
                "c": cmy[0], "m": cmy[1], "y": cmy[2],
            },
        }
        canvas = sim.render_stroke(canvas, stroke)
        yy += line_spacing
    return canvas


def _sample_centre(sim, canvas: torch.Tensor) -> torch.Tensor:
    """Sample a small patch at the canvas centre and return mean RGB."""
    cx = sim._cal.render.work_area_mm[0] / 2
    cy = sim._cal.render.work_area_mm[1] / 2
    px_x = int(cx * sim._dpi_x)
    px_y = int(cy * sim._dpi_y)
    px_x = min(max(px_x, 0), sim._W - 1)
    px_y = min(max(px_y, 0), sim._H - 1)
    r = 2
    y0 = max(px_y - r, 0)
    y1 = min(px_y + r + 1, sim._H)
    x0 = max(px_x - r, 0)
    x1 = min(px_x + r + 1, sim._W)
    return canvas[0, :, y0:y1, x0:x1].mean(dim=(1, 2))


def _fmt_rgb(rgb) -> str:
    """Format an RGB list/tuple for table display."""
    return f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})"


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

    all_de: List[float] = []
    for block_data in results.values():
        for entry in block_data.values():
            all_de.append(entry["delta_e"])

    cal_dict = session.cal.model_dump(mode="json")

    if all_de:
        mean_de = sum(all_de) / len(all_de)
        max_de = max(all_de)
        session.console.print(
            f"\nOverall  mean dE = {mean_de:.2f}   "
            f"max dE = {max_de:.2f}",
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


def _save_canvas_png(canvas: torch.Tensor, path: Path) -> None:
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

    Returns list of stroke dicts (tiny-length strokes -> isolated dots).
    """
    strokes = []
    c, m, y = color_cmy
    tiny = 0.01
    for row, z in enumerate(z_values):
        cy = spacing_mm * (row + 1)
        for col in range(n_repeats):
            cx = spacing_mm * (col + 1)
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
        if questionary.confirm(
            "Save unsaved changes?", default=True,
        ).ask():
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
