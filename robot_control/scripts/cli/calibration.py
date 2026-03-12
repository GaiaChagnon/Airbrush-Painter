"""Calibration mode -- persistent panels with interactive routines.

Wraps ``calibration/routines.py`` with a questionary-driven mode
selector and Rich-rendered results.  The bed-mesh routine uses
a persistent ``ProbeGridPanel`` that updates in place via
``console.clear()`` + re-render between each probe point.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robot_control.calibration import routines
from robot_control.calibration.measurement import (
    binary_search_z,
    get_float_input,
    get_yes_no,
    check_planarity,
)
from robot_control.configs.loader import MachineConfig, load_config
from robot_control.configs.printer_cfg import generate_printer_cfg
from robot_control.hardware.klipper_client import KlipperClient, KlipperShutdown
from robot_control.scripts.cli.widgets import info_panel, probe_grid_panel
from src.utils.fs import atomic_yaml_dump, load_yaml

if TYPE_CHECKING:
    from robot_control.scripts.cli.app import RobotApp

_FIRST_POINT_SAFE_MARGIN_MM = 2.0
_SUBSEQUENT_SAFE_MARGIN_MM = 0.25
_TEST_CIRCLE_RADIUS_MM = 3.0


# ======================================================================
# Persistent UI helpers for calibration
# ======================================================================


class _CalSession:
    """Lightweight session holder for calibration modes that need
    persistent rendering (clear + re-render before each prompt).
    """

    _MAX_LOG_LINES = 40

    def __init__(self, app: RobotApp) -> None:
        self.app = app
        self.console = app.console
        self._log_lines: list[str] = []
        self._top_panel: Panel | None = None

    def log(self, msg: str) -> None:
        self._log_lines.append(msg)
        if len(self._log_lines) > self._MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self._MAX_LOG_LINES :]

    def set_top_panel(self, panel: Panel | None) -> None:
        """Set the main panel shown at the top (e.g. probe grid)."""
        self._top_panel = panel

    def render_screen(self) -> None:
        """Clear terminal, render top panel + activity log, then return
        so a questionary prompt can appear below."""
        self.console.clear()
        if self._top_panel:
            self.console.print(self._top_panel)
        if self._log_lines:
            term_h = self.console.height or 40
            overhead = 20 if self._top_panel else 5
            n_visible = max(4, term_h - overhead)
            lines = self._log_lines[-n_visible:]
            parts: list[Text] = []
            for line in lines:
                try:
                    parts.append(Text.from_markup(line))
                except Exception:
                    parts.append(Text(line))
            content = Text("\n").join(parts)
            self.console.print(
                Panel(content, title="[bold]Activity Log[/]", border_style="dim")
            )


# ======================================================================
# Calibration routine registry
# ======================================================================

_ROUTINES: list[tuple[str, str]] = [
    ("Steps/mm calibration (X axis)", "steps_x"),
    ("Steps/mm calibration (Y axis)", "steps_y"),
    ("Z-height calibration", "z_heights"),
    ("Tool offset calibration", "tool_offset"),
    ("Speed calibration", "speed"),
    ("Endstop repeatability test", "endstops"),
    ("Endstop phase calibration", "endstop_phase"),
    ("Digital output: Pump refill servo", "servo_refill"),
    ("Digital output: Airbrush needle servo", "servo_needle"),
    ("Digital output: Air supply valve", "air_valve"),
    ("Bed mesh calibration", "bed_mesh"),
]

_NEEDS_HOMING = {
    "steps_x", "steps_y", "z_heights", "tool_offset", "speed",
    "endstops", "endstop_phase", "bed_mesh",
}


def run(app: RobotApp) -> None:
    """Entry point for calibration mode."""
    console = app.console
    config = app.config
    log = app.session_log

    console.clear()
    choices = [label for label, _ in _ROUTINES] + ["Back to main menu"]
    selected = questionary.select(
        "Select calibration routine:",
        choices=choices,
    ).ask()
    if selected is None or selected == "Back to main menu":
        return

    routine_key = ""
    for label, key in _ROUTINES:
        if label == selected:
            routine_key = key
            break

    log.log_action("calibration", "routine_select", routine_key)

    if not app.connection.is_connected():
        console.print("[yellow]Connecting to Klipper...[/]")
        if not app.no_config_write:
            app.connection.regenerate_printer_cfg(force=True)
        app.connection.connect()

    client = app.connection.client

    try:
        client.send_gcode("SET_IDLE_TIMEOUT TIMEOUT=3600", timeout=5.0)
    except Exception:
        pass

    try:
        client.send_gcode("BED_MESH_CLEAR", timeout=5.0)
    except Exception:
        pass

    needs_homing = routine_key in _NEEDS_HOMING
    if needs_homing:
        console.print("\n  Homing all axes...")
        try:
            client.send_gcode("G28\nM400", timeout=60.0)
            console.print("  [green]Homed.[/]")
        except Exception as exc:
            console.print(f"  [red]Homing failed: {exc}[/]")
            return

    try:
        if routine_key == "steps_x":
            _run_steps(app, client, config, "X")
        elif routine_key == "steps_y":
            _run_steps(app, client, config, "Y")
        elif routine_key == "z_heights":
            _run_z_heights(app, client, config)
        elif routine_key == "tool_offset":
            _run_tool_offset(app, client, config)
        elif routine_key == "speed":
            _run_speed(app, client, config)
        elif routine_key == "endstops":
            _run_endstops(app, client, config)
        elif routine_key == "endstop_phase":
            _run_endstop_phase(app, client, config)
        elif routine_key in ("servo_refill", "servo_needle", "air_valve"):
            _run_digital_output(app, client, config, routine_key)
        elif routine_key == "bed_mesh":
            _run_bed_mesh(app, client, config)

    except KeyboardInterrupt:
        console.print("\n[yellow]Calibration interrupted.[/]")
    except Exception as exc:
        console.print(f"\n[red]Calibration error: {exc}[/]")
        log.log_error(exc)
    finally:
        try:
            client.send_gcode(
                f"SET_IDLE_TIMEOUT TIMEOUT={config.motion.idle_timeout_s:.0f}",
                timeout=5.0,
            )
            client.send_gcode("M84", timeout=5.0)
        except Exception:
            pass

    console.print("[green]Calibration complete.[/]")


# ======================================================================
# Individual routine wrappers (simple routines: clear + result panel)
# ======================================================================


def _run_steps(
    app: RobotApp, client: KlipperClient, config: MachineConfig, axis: str
) -> None:
    app.console.clear()
    app.console.rule(f"[bold]Steps/mm Calibration -- {axis} Axis[/]")
    result = routines.calibrate_steps_per_mm(client, config, axis=axis)
    app.console.clear()
    app.console.print(
        info_panel(
            {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in result.items()},
            title=f"{axis} Calibration Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", f"steps_{axis.lower()}", str(result))
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_z_heights(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    app.console.clear()
    app.console.rule("[bold]Z-Height Calibration[/]")
    result = routines.calibrate_z_heights(client, config)
    app.console.clear()
    app.console.print(
        info_panel(
            {k: f"{v:.3f} mm" if isinstance(v, float) else str(v) for k, v in result.items()},
            title="Z-Height Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", "z_heights", str(result))
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_tool_offset(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    app.console.clear()
    app.console.rule("[bold]Tool Offset Calibration[/]")
    result = routines.calibrate_tool_offset(client, config)
    app.console.clear()
    app.console.print(
        info_panel(
            {k: f"{v:.3f}" if isinstance(v, float) else str(v) for k, v in result.items()},
            title="Tool Offset Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", "tool_offset", str(result))
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_speed(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    app.console.clear()
    app.console.rule("[bold]Speed Calibration[/]")
    result = routines.calibrate_speed(client, config)
    app.console.clear()
    app.console.print(
        info_panel(
            {k: str(v) for k, v in result.items()},
            title="Speed Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", "speed", str(result))
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_endstops(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    app.console.clear()
    app.console.rule("[bold]Endstop Repeatability Test[/]")
    cycles_str = questionary.text("Number of cycles [10]:", default="10").ask()
    cycles = int(cycles_str) if cycles_str and cycles_str.strip().isdigit() else 10
    result = routines.verify_endstops(client, config, cycles=cycles)

    table = Table(title=f"Endstop Results ({cycles} cycles)")
    table.add_column("Axis", justify="center")
    table.add_column("Mean", justify="right")
    table.add_column("StdDev", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Result", justify="center")

    for label in ("x", "y", "z"):
        stats = result[label]
        tol = 0.1 if label != "z" else 0.05
        ok = stats["range"] < tol
        color = "green" if ok else "red"
        table.add_row(
            label.upper(),
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"[{color}]{stats['range']:.4f}[/]",
            f"[{color}]{'PASS' if ok else 'FAIL'}[/]",
        )

    app.console.clear()
    app.console.print(table)
    overall = "[green]PASS[/]" if result["passed"] else "[red]FAIL[/]"
    app.console.print(f"\n  Overall: {overall}")
    app.session_log.log_action(
        "calibration", "endstops", f"passed={result['passed']}"
    )
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_endstop_phase(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    app.console.clear()
    app.console.rule("[bold]Endstop Phase Calibration[/]")
    result = routines.calibrate_endstop_phase(client, config)
    app.console.clear()
    app.console.print(
        info_panel(
            {
                "Homes": str(result.get("homes", "?")),
                "Output": result.get("klipper_output", "(none)"),
            },
            title="Endstop Phase Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", "endstop_phase")
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def _run_digital_output(
    app: RobotApp,
    client: KlipperClient,
    config: MachineConfig,
    key: str,
) -> None:
    name_map = {
        "servo_refill": "servo_pump_refill",
        "servo_needle": "servo_airbrush_needle",
        "air_valve": "air_valve",
    }
    output_name = name_map[key]
    app.console.clear()
    app.console.rule(f"[bold]Digital Output Test: {output_name}[/]")
    result = routines.test_digital_output(client, config, output_name)
    app.console.clear()
    app.console.print(
        info_panel(
            {k: str(v) for k, v in result.items()},
            title="Digital Output Result",
            border_style="green",
        )
    )
    app.session_log.log_action("calibration", "digital_output", output_name)
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


# ======================================================================
# Bed mesh calibration (persistent grid panel)
# ======================================================================


def _run_bed_mesh(
    app: RobotApp, client: KlipperClient, config: MachineConfig
) -> None:
    """Enhanced bed mesh calibration with persistent grid visualisation.

    The probe-grid panel stays at the top of the terminal.  After each
    probe point, ``console.clear()`` + re-render places the updated
    grid panel at the top so it never scrolls away.
    """
    cs = _CalSession(app)
    console = app.console

    if config.bed_mesh is None:
        console.print("[red]bed_mesh section missing from machine.yaml[/]")
        return

    bm = config.bed_mesh
    config_path = Path(__file__).resolve().parents[2] / "configs" / "machine.yaml"

    jobs_path = Path(__file__).resolve().parents[2] / "configs" / "jobs.yaml"
    jobs = load_yaml(jobs_path)
    paper = jobs.get("paper", {})
    pen_tracer = jobs.get("pen_tracer", {})
    z_contact = pen_tracer.get("z_contact_mm")
    if z_contact is None:
        console.print("[red]pen_tracer.z_contact_mm not found in jobs.yaml[/]")
        return
    z_contact = float(z_contact)

    console.clear()
    console.print(
        info_panel(
            {
                "z_contact_mm": f"{z_contact:.3f} (from jobs.yaml)",
                "Grid": f"{bm.probe_count[0]}x{bm.probe_count[1]}",
            },
            title="Bed Mesh Setup",
        )
    )

    use_full_canvas = questionary.confirm(
        "Probe full canvas area (instead of paper bounds)?",
        default=False,
    ).ask()
    if use_full_canvas is None:
        return

    if use_full_canvas:
        mesh_min = bm.mesh_min
        mesh_max = bm.mesh_max
        bounds_label = "full canvas"
    else:
        origin = paper.get("origin_mm", [25.0, 25.0])
        size = paper.get("size_mm", [297.0, 210.0])
        mesh_min = (float(origin[0]), float(origin[1]))
        mesh_max = (float(origin[0] + size[0]), float(origin[1] + size[1]))
        bounds_label = "paper (jobs.yaml)"

    probe_count = bm.probe_count
    nx, ny = probe_count
    first_z_start = z_contact - _FIRST_POINT_SAFE_MARGIN_MM

    # Build probe grid
    probe_points: list[tuple[float, float]] = []
    for yi in range(ny):
        y = (
            mesh_min[1]
            if ny == 1
            else mesh_min[1] + yi * (mesh_max[1] - mesh_min[1]) / (ny - 1)
        )
        for xi in range(nx):
            x = (
                mesh_min[0]
                if nx == 1
                else mesh_min[0] + xi * (mesh_max[0] - mesh_min[0]) / (nx - 1)
            )
            probe_points.append((round(x, 2), round(y, 2)))

    total = len(probe_points)
    results_for_grid: list[float | None] = [None] * total

    cs.set_top_panel(
        probe_grid_panel(probe_points, probe_count, title="Probe Points (before probing)")
    )
    cs.log(f"  Bounds: {bounds_label}")
    cs.log(f"  Grid: {nx}x{ny} = {total} points")
    cs.log(
        f"  Area: ({mesh_min[0]:.1f}, {mesh_min[1]:.1f}) -> "
        f"({mesh_max[0]:.1f}, {mesh_max[1]:.1f}) mm"
    )
    cs.log(f"  First start Z: {first_z_start:.3f} mm")
    cs.log("  Place paper on the glass surface.")
    cs.render_screen()

    questionary.press_any_key_to_continue("Press any key to begin...").ask()

    try:
        client.send_gcode("BED_MESH_CLEAR", timeout=5.0)
    except Exception:
        pass

    cs.log("  Homing all axes...")
    client.send_gcode("G28\nM400", timeout=60.0)
    cs.log("  [green]Homed.[/]")

    tc = config.get_tool("pen")
    f_travel = tc.travel_feed_mm_s * 60.0
    f_plunge = tc.plunge_feed_mm_s * 60.0
    z_safe = bm.horizontal_move_z

    z_start = first_z_start
    contact_zs: list[float] = []
    final_xys: list[tuple[float, float]] = []

    for idx, (px, py) in enumerate(probe_points):
        cs.set_top_panel(
            probe_grid_panel(
                probe_points,
                probe_count,
                results=results_for_grid,
                title=f"Probing Point {idx + 1}/{total}",
            )
        )
        cs.log(f"  [bold]Point {idx + 1}/{total}[/]  X={px:.1f}  Y={py:.1f}  start Z={z_start:.3f}")
        cs.render_screen()

        client.send_gcode(f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0)
        client.send_gcode(
            f"G0 X{px:.2f} Y{py:.2f} F{f_travel:.0f}\nM400", timeout=10.0
        )
        client.send_gcode(f"G1 Z{z_start:.3f} F{f_plunge:.0f}\nM400", timeout=10.0)

        current_z, current_x, current_y = z_start, px, py

        while True:
            raw = questionary.text(
                f"  X={current_x:.1f} Y={current_y:.1f} Z={current_z:.3f} >",
            ).ask()
            if raw is None:
                raise KeyboardInterrupt
            raw = raw.strip().lower()

            if raw == "accept":
                cs.log(f"  [green]Accepted Z = {current_z:.3f} mm[/]")
                break

            if raw == "circle":
                cs.log("  Drawing test circle...")
                routines._draw_test_circle(
                    client, current_x, current_y, current_z, config
                )
                client.send_gcode(
                    f"G1 Z{current_z:.3f} F{f_plunge:.0f}\nM400", timeout=10.0
                )
                continue

            xy_jog = {"l": (-5, 0), "r": (5, 0), "f": (0, 5), "b": (0, -5)}
            if raw in xy_jog:
                dx, dy = xy_jog[raw]
                new_x = max(0, min(current_x + dx, config.work_area.x))
                new_y = max(0, min(current_y + dy, config.work_area.y))
                client.send_gcode(
                    f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0
                )
                client.send_gcode(
                    f"G0 X{new_x:.2f} Y{new_y:.2f} F{f_travel:.0f}\nM400",
                    timeout=10.0,
                )
                client.send_gcode(
                    f"G1 Z{current_z:.3f} F{f_plunge:.0f}\nM400", timeout=10.0
                )
                current_x, current_y = new_x, new_y
                continue

            try:
                delta = float(raw)
            except ValueError:
                cs.log("  [dim]Type number, l/r/f/b, 'circle', or 'accept'[/]")
                cs.render_screen()
                continue

            z_min = config.work_area.soft_limits.z_min
            z_max = config.work_area.soft_limits.z_max_with_overtravel
            new_z = max(z_min, min(current_z + delta, z_max))
            client.send_gcode(
                f"G1 Z{new_z:.3f} F{f_plunge:.0f}\nM400", timeout=10.0
            )
            current_z = new_z

        contact_zs.append(current_z)
        final_xys.append((current_x, current_y))
        results_for_grid[idx] = current_z
        z_start = current_z - _SUBSEQUENT_SAFE_MARGIN_MM
        app.session_log.log_action(
            "calibration",
            "probe_point",
            f"({current_x:.1f},{current_y:.1f}) Z={current_z:.3f}",
        )

    client.send_gcode(f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0)

    # ------ Results ------
    mean_z = sum(contact_zs) / len(contact_zs)
    offsets_flat = [z - mean_z for z in contact_zs]
    offsets_2d: list[list[float]] = []
    for row_idx in range(ny):
        row = offsets_flat[row_idx * nx : (row_idx + 1) * nx]
        offsets_2d.append([round(v, 6) for v in row])

    points_xyz = [(fx, fy, cz) for (fx, fy), cz in zip(final_xys, contact_zs)]
    planarity = check_planarity(points_xyz, tolerance=0.15)

    cs.set_top_panel(
        probe_grid_panel(
            probe_points,
            probe_count,
            results=contact_zs,
            mean_z=mean_z,
            title="Bed Mesh Results",
        )
    )

    planarity_str = "[green]PASS[/]" if planarity.is_planar else "[yellow]WARNING[/]"
    cs.log(f"  Mean contact Z: {mean_z:.3f} mm")
    cs.log(
        f"  Offset range: [{min(offsets_flat):.3f}, {max(offsets_flat):.3f}] mm"
    )
    cs.log(f"  Max plane residual: {planarity.max_residual_mm:.4f} mm")
    cs.log(f"  Planarity: {planarity_str}")

    if not planarity.is_planar:
        cs.log("[yellow]Surface may be flexing or not rigid.[/]")
        for i, ((fx, fy), res) in enumerate(
            zip(final_xys, planarity.residuals)
        ):
            color = "red" if abs(res) > 0.15 else "green"
            cs.log(
                f"    Point {i + 1} ({fx:.0f}, {fy:.0f}): "
                f"[{color}]{res:+.4f} mm[/]"
            )

    cs.render_screen()

    if app.confirm_dangerous(
        "Save mesh to machine.yaml and regenerate printer.cfg"
    ):
        routines._save_bed_mesh_to_config(
            config_path, offsets_2d, mean_z, mesh_min, mesh_max
        )
        cs.log(f"  Updated machine.yaml (pen_work_mm = {mean_z:.3f})")

        routines._save_z_contact_to_jobs(mean_z)
        cs.log(f"  Updated jobs.yaml (z_contact_mm = {mean_z:.3f})")

        updated_cfg = load_config(config_path)
        printer_cfg_path = Path.home() / "printer.cfg"
        if printer_cfg_path.exists():
            backup = printer_cfg_path.with_suffix(".cfg.bak")
            printer_cfg_path.rename(backup)
            cs.log(f"  Backed up printer.cfg -> {backup}")
        printer_cfg_text = generate_printer_cfg(updated_cfg)
        printer_cfg_path.write_text(printer_cfg_text)
        cs.log(f"  Wrote printer.cfg to {printer_cfg_path}")
        cs.log("  Restart Klipper to load the new mesh profile.")
        cs.log("  Drawing scripts will run: BED_MESH_PROFILE LOAD=default")
    else:
        cs.log("  Mesh NOT saved.")

    cs.render_screen()
    questionary.press_any_key_to_continue("Press any key to continue...").ask()

    app.session_log.log_action(
        "calibration",
        "bed_mesh_complete",
        f"mean_z={mean_z:.3f} planarity={'PASS' if planarity.is_planar else 'WARNING'}",
    )
