"""Guided calibration routines.

Each routine is an interactive function that:
    1. Executes a test pattern.
    2. Prompts the user for measurements.
    3. Calculates corrections.
    4. Optionally updates the config file.

All routines accept a ``KlipperClient`` and ``MachineConfig`` and return
a results dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

from robot_control.calibration import patterns
from robot_control.calibration.measurement import (
    MeasurementResult,
    PlanarityResult,
    binary_search_z,
    calculate_new_rotation_distance,
    calculate_steps_correction,
    calculate_tool_offset,
    check_planarity,
    format_calibration_summary,
    get_choice,
    get_float_input,
    get_yes_no,
)
from robot_control.configs.loader import MachineConfig
from robot_control.hardware.job_executor import JobExecutor
from robot_control.hardware.klipper_client import KlipperClient
from src.utils.fs import atomic_yaml_dump, load_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steps per mm (rotation_distance) calibration
# ---------------------------------------------------------------------------


def calibrate_steps_per_mm(
    client: KlipperClient,
    config: MachineConfig,
    axis: str = "X",
    distance_mm: float = 100.0,
) -> dict[str, Any]:
    """Calibrate steps/mm for an axis by drawing a ruler and measuring.

    Formula::

        new_rotation_distance = old * (commanded / measured)

    Parameters
    ----------
    client : KlipperClient
        Connected client.
    config : MachineConfig
        Current config (``steppers.xy_rotation_distance`` used as old value).
    axis : str
        ``"X"`` or ``"Y"``.
    distance_mm : float
        Ruler length to draw (mm).

    Returns
    -------
    dict
        ``axis``, ``commanded``, ``measured``, ``correction_factor``,
        ``old_rotation_distance``, ``new_rotation_distance``.
    """
    executor = JobExecutor(client, config)

    logger.info("Starting steps/mm calibration for %s axis", axis)
    print(f"\n{'='*50}")
    print(f"  STEPS/MM CALIBRATION -- {axis} AXIS")
    print(f"{'='*50}\n")

    # Draw ruler pattern
    if axis.upper() == "X":
        ops = patterns.ruler_x(length=distance_mm)
    else:
        ops = patterns.ruler_y(length=distance_mm)

    print(f"Drawing {distance_mm} mm ruler on {axis} axis...")
    executor.run_interactive(ops)

    # Get measurement
    measured = get_float_input(
        f"Measure the actual length with a ruler (expected {distance_mm} mm)",
    )

    result = calculate_steps_correction(distance_mm, measured)
    old_rd = config.steppers.xy_rotation_distance
    new_rd = calculate_new_rotation_distance(old_rd, distance_mm, measured)

    summary = {
        "axis": axis,
        "commanded": distance_mm,
        "measured": measured,
        "error_mm": result.error,
        "error_percent": result.error_percent,
        "correction_factor": result.correction_factor,
        "old_rotation_distance": old_rd,
        "new_rotation_distance": new_rd,
    }

    print(format_calibration_summary(summary))

    if get_yes_no("Update config with new rotation_distance?"):
        print(f"  New xy_rotation_distance: {new_rd:.4f}")
        print("  (Update machine.yaml manually with this value)")

    return summary


# ---------------------------------------------------------------------------
# Z seesaw calibration
# ---------------------------------------------------------------------------


def calibrate_z_heights(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """Interactively find pen and airbrush work Z heights.

    Uses a binary search: move Z incrementally, ask user if the tool is
    touching the paper.

    Returns
    -------
    dict
        ``pen_work_z``, ``airbrush_work_z``, ``travel_z``.
    """
    logger.info("Starting Z-height calibration")
    print(f"\n{'='*50}")
    print("  Z-HEIGHT (SEESAW) CALIBRATION")
    print(f"{'='*50}\n")

    # Home first
    print("Homing X Y...")
    client.send_gcode("G28 X Y\nM400", timeout=30.0)

    # Move to test position (canvas centre)
    cx = config.canvas.offset_x_mm + config.canvas.width_mm / 2
    cy = config.canvas.offset_y_mm + config.canvas.height_mm / 2
    tc = config.get_tool("pen")
    f_travel = tc.travel_feed_mm_s * 60.0
    client.send_gcode(f"G0 X{cx:.1f} Y{cy:.1f} F{f_travel:.0f}\nM400")

    print("\nPlace paper on the canvas.\n")

    # Pen calibration
    print("--- Pen Z calibration ---")
    print("The Z axis will move incrementally.  Answer when pen touches paper.")

    def pen_prompt(z: float) -> bool:
        f_plunge = tc.plunge_feed_mm_s * 60.0
        client.send_gcode(f"G0 Z{z:.3f} F{f_plunge:.0f}\nM400")
        return get_yes_no(f"Z = {z:.3f} mm -- is the pen touching paper?")

    z_min, z_max = 0.0, config.work_area.z
    pen_z = binary_search_z(pen_prompt, z_min, z_max, tolerance=0.05)
    print(f"  Pen work Z: {pen_z:.3f} mm")

    # Raise to travel
    client.send_gcode(
        f"G0 Z{config.z_states.travel_mm:.1f} F{tc.plunge_feed_mm_s * 60:.0f}\nM400",
    )

    # Airbrush calibration
    print("\n--- Airbrush Z calibration ---")
    print("Same procedure for the airbrush side of the seesaw.")

    def airbrush_prompt(z: float) -> bool:
        f_plunge = tc.plunge_feed_mm_s * 60.0
        client.send_gcode(f"G0 Z{z:.3f} F{f_plunge:.0f}\nM400")
        return get_yes_no(
            f"Z = {z:.3f} mm -- is the airbrush at correct spray height?",
        )

    ab_z = binary_search_z(airbrush_prompt, z_min, z_max, tolerance=0.05)
    print(f"  Airbrush work Z: {ab_z:.3f} mm")

    travel_z = (pen_z + ab_z) / 2.0

    summary = {
        "pen_work_z": pen_z,
        "airbrush_work_z": ab_z,
        "travel_z": travel_z,
    }
    print(format_calibration_summary(summary))
    return summary


# ---------------------------------------------------------------------------
# Tool offset calibration
# ---------------------------------------------------------------------------


def calibrate_tool_offset(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """Measure XY offset between pen and airbrush via crosshair patterns.

    Returns
    -------
    dict
        ``offset_x``, ``offset_y``.
    """
    executor = JobExecutor(client, config)

    logger.info("Starting tool-offset calibration")
    print(f"\n{'='*50}")
    print("  TOOL OFFSET CALIBRATION")
    print(f"{'='*50}\n")

    # Draw crosshair with pen
    print("Step 1: Drawing crosshair with PEN...")
    pen_ops = patterns.cross(size_mm=30.0, tool="pen")
    executor.run_interactive(pen_ops)

    # Draw crosshair with airbrush at same commanded position
    print("Step 2: Drawing crosshair with AIRBRUSH at same position...")
    ab_ops = patterns.cross(size_mm=30.0, tool="airbrush")
    executor.run_interactive(ab_ops)

    print("\nMeasure the offset between the two crosshair centres.")
    pen_x = get_float_input("Pen crosshair centre X (mm)", default=0.0)
    pen_y = get_float_input("Pen crosshair centre Y (mm)", default=0.0)
    ab_x = get_float_input("Airbrush crosshair centre X (mm)", default=0.0)
    ab_y = get_float_input("Airbrush crosshair centre Y (mm)", default=0.0)

    ox, oy = calculate_tool_offset(pen_x, pen_y, ab_x, ab_y)

    summary: dict[str, Any] = {"offset_x": ox, "offset_y": oy}
    print(format_calibration_summary(summary))

    if get_yes_no("Update airbrush xy_offset_mm in config?"):
        print(f"  New airbrush xy_offset_mm: [{ox:.3f}, {oy:.3f}]")
        print("  (Update machine.yaml manually with these values)")

    return summary


# ---------------------------------------------------------------------------
# Speed calibration
# ---------------------------------------------------------------------------


def calibrate_speed(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """Draw lines at increasing speeds; user picks the best.

    Returns
    -------
    dict
        ``recommended_speed_mm_s``.
    """
    executor = JobExecutor(client, config)

    logger.info("Starting speed calibration")
    print(f"\n{'='*50}")
    print("  SPEED CALIBRATION")
    print(f"{'='*50}\n")

    speeds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
    print(f"Drawing lines at speeds: {speeds} mm/s")
    ops = patterns.speed_test(speeds=speeds)
    executor.run_interactive(ops)

    print("\nExamine the drawn lines (numbered bottom to top).")
    idx = int(
        get_float_input(
            f"Which line number (1-{len(speeds)}) is the highest quality?",
            default=4.0,
        ),
    )
    idx = max(1, min(idx, len(speeds)))
    recommended = speeds[idx - 1]

    summary: dict[str, Any] = {"recommended_speed_mm_s": recommended}
    print(format_calibration_summary(summary))
    return summary


# ---------------------------------------------------------------------------
# Endstop verification
# ---------------------------------------------------------------------------


def verify_endstops(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """Home, jog away, re-home, verify positions match.

    Uses ``G28 X Y`` which works under cartesian kinematics.  Under the
    current config, X homes to 0 (endstop at min) and Y homes to
    ``work_area.y`` (endstop at max, ``homing_positive_dir: True``).

    Endstop keys in the Klipper object model are ``stepper_x`` and
    ``stepper_y`` (not ``manual_stepper`` names).

    Returns
    -------
    dict
        ``passed``, ``home1``, ``home2``, ``delta``.
    """
    logger.info("Starting endstop verification")
    print(f"\n{'='*50}")
    print("  ENDSTOP VERIFICATION")
    print(f"{'='*50}\n")

    print("Home #1...")
    client.send_gcode("G28 X Y\nM400", timeout=30.0)
    pos1 = client.get_position()
    print(f"  Position after home #1: X={pos1.x:.3f} Y={pos1.y:.3f}")

    # Jog away
    tc = config.get_tool("pen")
    f_val = tc.travel_feed_mm_s * 60.0
    client.send_gcode(f"G0 X50 Y50 F{f_val:.0f}\nM400")

    print("Home #2...")
    client.send_gcode("G28 X Y\nM400", timeout=30.0)
    pos2 = client.get_position()
    print(f"  Position after home #2: X={pos2.x:.3f} Y={pos2.y:.3f}")

    dx = abs(pos2.x - pos1.x)
    dy = abs(pos2.y - pos1.y)
    passed = dx < 0.1 and dy < 0.1

    summary: dict[str, Any] = {
        "passed": passed,
        "home1_x": pos1.x,
        "home1_y": pos1.y,
        "home2_x": pos2.x,
        "home2_y": pos2.y,
        "delta_x": dx,
        "delta_y": dy,
    }
    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}  (delta X={dx:.3f}, Y={dy:.3f})")
    print(format_calibration_summary(summary))
    return summary


# ---------------------------------------------------------------------------
# Bed mesh (surface leveling) calibration
# ---------------------------------------------------------------------------

_TEST_CIRCLE_RADIUS_MM = 3.0
_PLANARITY_TOLERANCE_MM = 0.15
# First point starts this far BELOW z_contact (pen further from paper)
_FIRST_POINT_SAFE_MARGIN_MM = 2.0
# Subsequent points start this far below the previous accepted Z
_SUBSEQUENT_SAFE_MARGIN_MM = 0.25


def _build_probe_grid(
    mesh_min: tuple[float, float],
    mesh_max: tuple[float, float],
    probe_count: tuple[int, int],
) -> list[tuple[float, float]]:
    """Compute the XY probe points in Klipper order (Y rows, X cols).

    Returns points sorted by Y ascending then X ascending, matching the
    row order expected by the ``[bed_mesh default]`` save block.
    """
    nx, ny = probe_count
    points: list[tuple[float, float]] = []
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
                else mesh_min[0]
                + xi * (mesh_max[0] - mesh_min[0]) / (nx - 1)
            )
            points.append((round(x, 2), round(y, 2)))
    return points


def _draw_test_circle(
    client: KlipperClient,
    cx: float,
    cy: float,
    z_contact: float,
    config: MachineConfig,
) -> None:
    """Draw a small circle at the current probe point at *z_contact*.

    Pen is lowered to *z_contact*, a 3 mm radius circle is drawn via
    G2 arcs, then the pen is retracted.
    """
    tc = config.get_tool("pen")
    f_draw = tc.feed_mm_s * 60.0
    f_travel = tc.travel_feed_mm_s * 60.0
    f_plunge = tc.plunge_feed_mm_s * 60.0
    r = _TEST_CIRCLE_RADIUS_MM
    z_safe = config.bed_mesh.horizontal_move_z if config.bed_mesh else 75.0

    # Move to circle start (right side of the circle)
    client.send_gcode(
        f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0,
    )
    client.send_gcode(
        f"G0 X{cx + r:.2f} Y{cy:.2f} F{f_travel:.0f}\nM400", timeout=10.0,
    )
    # Plunge to contact
    client.send_gcode(
        f"G1 Z{z_contact:.3f} F{f_plunge:.0f}\nM400", timeout=10.0,
    )
    # Two semicircle arcs
    client.send_gcode(
        f"G2 X{cx - r:.2f} Y{cy:.2f} I{-r:.2f} J0 F{f_draw:.0f}",
        timeout=10.0,
    )
    client.send_gcode(
        f"G2 X{cx + r:.2f} Y{cy:.2f} I{r:.2f} J0 F{f_draw:.0f}",
        timeout=10.0,
    )
    # Retract
    client.send_gcode(
        f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0,
    )
    # Return to probe point centre
    client.send_gcode(
        f"G0 X{cx:.2f} Y{cy:.2f} F{f_travel:.0f}\nM400", timeout=10.0,
    )


_XY_JOG_STEP_MM = 5.0


def _probe_single_point(
    client: KlipperClient,
    config: MachineConfig,
    idx: int,
    total: int,
    x: float,
    y: float,
    z_start: float,
) -> tuple[float, float, float]:
    """Interactive Z-probing at a single point.

    The user types numeric values to jog Z, ``circle`` to draw a test
    circle, XY jog commands to nudge the position, or ``accept`` to
    record the current Z and move on.

    Parameters
    ----------
    x, y : float
        Initial XY probe position in mm.

    Returns
    -------
    tuple[float, float, float]
        ``(accepted_z, final_x, final_y)`` -- the contact Z and the
        (possibly nudged) XY position.
    """
    tc = config.get_tool("pen")
    f_plunge = tc.plunge_feed_mm_s * 60.0
    f_travel = tc.travel_feed_mm_s * 60.0
    z_min = config.work_area.soft_limits.z_min
    z_max = config.work_area.soft_limits.z_max_with_overtravel
    x_max = config.work_area.x
    y_max = config.work_area.y
    current_z = z_start
    current_x = x
    current_y = y

    xy_jog_aliases: dict[str, tuple[float, float]] = {
        "l":       (-_XY_JOG_STEP_MM, 0.0),
        "left":    (-_XY_JOG_STEP_MM, 0.0),
        "r":       (_XY_JOG_STEP_MM, 0.0),
        "right":   (_XY_JOG_STEP_MM, 0.0),
        "f":       (0.0, _XY_JOG_STEP_MM),
        "forward": (0.0, _XY_JOG_STEP_MM),
        "b":       (0.0, -_XY_JOG_STEP_MM),
        "back":    (0.0, -_XY_JOG_STEP_MM),
    }

    print(f"\n  --- Point {idx + 1}/{total}  X={current_x:.1f}  Y={current_y:.1f} ---")
    print(f"  Starting Z: {current_z:.3f} mm")
    print("  Commands:")
    print("    <number>       Jog Z by that amount (e.g. +0.1, -0.05)")
    print("    l / r          Jog X by -5 / +5 mm  (left / right)")
    print("    f / b          Jog Y by +5 / -5 mm  (forward / back)")
    print("    circle         Draw a 3 mm test circle at current Z")
    print("    accept         Record this Z and move to next point")

    while True:
        raw = input(
            f"  X={current_x:.1f} Y={current_y:.1f} Z={current_z:.3f} > "
        ).strip().lower()

        if raw == "accept":
            print(f"  Accepted Z = {current_z:.3f} mm  "
                  f"at X={current_x:.1f} Y={current_y:.1f}")
            return current_z, current_x, current_y

        if raw == "circle":
            print("  Drawing test circle...")
            _draw_test_circle(client, current_x, current_y, current_z, config)
            client.send_gcode(
                f"G1 Z{current_z:.3f} F{f_plunge:.0f}\nM400", timeout=10.0,
            )
            continue

        if raw in xy_jog_aliases:
            dx, dy = xy_jog_aliases[raw]
            new_x = max(0.0, min(current_x + dx, x_max))
            new_y = max(0.0, min(current_y + dy, y_max))
            # Retract, move, re-descend
            z_safe = (
                config.bed_mesh.horizontal_move_z
                if config.bed_mesh else 75.0
            )
            client.send_gcode(
                f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400",
                timeout=10.0,
            )
            client.send_gcode(
                f"G0 X{new_x:.2f} Y{new_y:.2f} F{f_travel:.0f}\nM400",
                timeout=10.0,
            )
            client.send_gcode(
                f"G1 Z{current_z:.3f} F{f_plunge:.0f}\nM400",
                timeout=10.0,
            )
            current_x = new_x
            current_y = new_y
            continue

        try:
            delta = float(raw)
        except ValueError:
            print("  Invalid input. Type a number, l/r/f/b, 'circle', "
                  "or 'accept'.")
            continue

        new_z = current_z + delta
        new_z = max(z_min, min(new_z, z_max))
        client.send_gcode(
            f"G1 Z{new_z:.3f} F{f_plunge:.0f}\nM400", timeout=10.0,
        )
        current_z = new_z


def _save_bed_mesh_to_config(
    config_path: Path,
    calibrated_points: list[list[float]],
    new_pen_work_mm: float,
    mesh_min: tuple[float, float],
    mesh_max: tuple[float, float],
) -> None:
    """Write calibrated mesh offsets and bounds to machine.yaml.

    Also updates ``pen_work_mm`` to the mean contact Z so the mesh
    offsets are centred around zero.

    Performs a read-modify-write on the raw YAML dict.
    """
    data = load_yaml(config_path)

    if "bed_mesh" not in data:
        data["bed_mesh"] = {}
    data["bed_mesh"]["calibrated_points"] = calibrated_points
    data["bed_mesh"]["mesh_min"] = [round(v, 1) for v in mesh_min]
    data["bed_mesh"]["mesh_max"] = [round(v, 1) for v in mesh_max]

    data["z_states"]["pen_work_mm"] = round(new_pen_work_mm, 3)

    atomic_yaml_dump(data, config_path)
    logger.info(
        "Saved bed mesh (%d points), pen_work_mm=%.3f, "
        "bounds=(%.1f,%.1f)-(%.1f,%.1f) to %s",
        sum(len(row) for row in calibrated_points),
        new_pen_work_mm,
        mesh_min[0], mesh_min[1], mesh_max[0], mesh_max[1],
        config_path,
    )


def _save_z_contact_to_jobs(new_z_contact_mm: float) -> None:
    """Update ``pen_tracer.z_contact_mm`` in ``jobs.yaml``.

    Performs a read-modify-write so other fields remain untouched.
    """
    jobs_path = (
        Path(__file__).resolve().parent.parent / "configs" / "jobs.yaml"
    )
    data = load_yaml(jobs_path)

    if "pen_tracer" not in data:
        data["pen_tracer"] = {}
    data["pen_tracer"]["z_contact_mm"] = round(new_z_contact_mm, 3)

    atomic_yaml_dump(data, jobs_path)
    logger.info(
        "Saved z_contact_mm=%.3f to %s", new_z_contact_mm, jobs_path,
    )


class _JobsPenConfig(NamedTuple):
    """Subset of jobs.yaml needed by bed-mesh calibration."""

    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    z_contact_mm: float


def _load_jobs_pen_config() -> _JobsPenConfig:
    """Load paper bounds and ``z_contact_mm`` from ``jobs.yaml``.

    Returns
    -------
    _JobsPenConfig
        ``mesh_min/mesh_max`` covering the paper area (origin to
        origin+size) without margin inset, plus the current
        ``z_contact_mm`` pen-touch height.

    Raises
    ------
    KeyError
        If ``pen_tracer.z_contact_mm`` is missing from the YAML.
    """
    jobs_path = (
        Path(__file__).resolve().parent.parent / "configs" / "jobs.yaml"
    )
    jobs = load_yaml(jobs_path)

    paper = jobs.get("paper", {})
    origin = paper.get("origin_mm", [25.0, 25.0])
    size = paper.get("size_mm", [297.0, 210.0])

    mesh_min = (float(origin[0]), float(origin[1]))
    mesh_max = (float(origin[0] + size[0]), float(origin[1] + size[1]))

    pen_tracer = jobs.get("pen_tracer", {})
    z_contact = pen_tracer.get("z_contact_mm")
    if z_contact is None:
        raise KeyError(
            "pen_tracer.z_contact_mm not found in jobs.yaml -- "
            "this is required to determine the safe starting Z."
        )

    return _JobsPenConfig(mesh_min, mesh_max, float(z_contact))


def calibrate_bed_mesh(
    client: KlipperClient,
    config: MachineConfig,
    config_path: Path | None = None,
    planarity_tolerance: float = _PLANARITY_TOLERANCE_MM,
    use_full_canvas: bool = False,
) -> dict[str, Any]:
    """Manually probe a grid of points to build a Klipper bed-mesh profile.

    The reference contact height is read from ``jobs.yaml``'s
    ``pen_tracer.z_contact_mm``.  The first probe point starts
    ``_FIRST_POINT_SAFE_MARGIN_MM`` (2 mm) below that value (pen
    further from paper on the seesaw), so the pen cannot crash on
    initial approach.  Every subsequent point starts
    ``_SUBSEQUENT_SAFE_MARGIN_MM`` (0.25 mm) below the previously
    accepted contact Z.

    At each point the user jogs Z until the pen just touches the paper,
    optionally draws a test circle to verify contact, then accepts.
    After all points are probed the routine:

    1. Computes the mean contact Z and updates ``pen_work_mm`` in
       ``machine.yaml`` **and** ``z_contact_mm`` in ``jobs.yaml``.
    2. Computes per-point mesh offsets (contact - mean).
    3. Runs a planarity check (least-squares plane fit).
    4. Writes the mesh data to ``machine.yaml``.
    5. Regenerates ``printer.cfg`` with the saved mesh profile.

    Parameters
    ----------
    client : KlipperClient
        Connected and homed client.
    config : MachineConfig
        Current machine configuration (must include ``bed_mesh``).
    config_path : Path | None
        Explicit path to ``machine.yaml``.  ``None`` uses the default
        location alongside ``loader.py``.
    planarity_tolerance : float
        Max acceptable plane-fit residual in mm (default 0.15).
    use_full_canvas : bool
        When ``False`` (default), probe within the paper bounds from
        ``jobs.yaml``.  When ``True``, probe the full canvas area
        from ``machine.yaml``'s ``bed_mesh.mesh_min/mesh_max``.

    Returns
    -------
    dict
        ``mean_contact_z``, ``offsets``, ``planarity``, ``max_residual``.

    Raises
    ------
    RuntimeError
        If ``config.bed_mesh`` is not configured.
    """
    if config.bed_mesh is None:
        raise RuntimeError(
            "bed_mesh section missing from machine.yaml.  "
            "Add it before running bed-mesh calibration."
        )

    bm = config.bed_mesh
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent / "configs" / "machine.yaml"
        )

    jobs_pen = _load_jobs_pen_config()
    z_contact = jobs_pen.z_contact_mm

    # Determine probe bounds
    if use_full_canvas:
        mesh_min = bm.mesh_min
        mesh_max = bm.mesh_max
        bounds_label = "full canvas"
    else:
        mesh_min = jobs_pen.mesh_min
        mesh_max = jobs_pen.mesh_max
        bounds_label = "paper (jobs.yaml)"

    probe_count = bm.probe_count

    # First point: approach from z_contact minus a safe margin (pen well
    # above paper).  Subsequent points start just above the previously
    # accepted contact Z.  Lower Z = pen further from paper on the seesaw.
    first_z_start = z_contact - _FIRST_POINT_SAFE_MARGIN_MM

    logger.info("Starting bed-mesh calibration")
    print(f"\n{'='*60}")
    print("  BED MESH CALIBRATION (surface leveling)")
    print(f"{'='*60}")
    print(f"  Bounds:       {bounds_label}")
    print(f"  Grid:         {probe_count[0]}x{probe_count[1]}")
    print(f"  Area:         ({mesh_min[0]:.1f}, {mesh_min[1]:.1f}) -> "
          f"({mesh_max[0]:.1f}, {mesh_max[1]:.1f}) mm")
    print(f"  z_contact_mm: {z_contact:.3f}  (from jobs.yaml)")
    print(f"  First start:  {first_z_start:.3f}  "
          f"({_FIRST_POINT_SAFE_MARGIN_MM} mm safety margin)")
    print(f"  Subsequent:   accepted_z - {_SUBSEQUENT_SAFE_MARGIN_MM} mm")
    print()

    probe_points = _build_probe_grid(mesh_min, mesh_max, probe_count)
    total = len(probe_points)

    tc = config.get_tool("pen")
    f_travel = tc.travel_feed_mm_s * 60.0
    f_plunge = tc.plunge_feed_mm_s * 60.0
    z_safe = bm.horizontal_move_z

    z_start = first_z_start

    print("  Place paper on the glass surface.")
    print("  The pen will move to each point; adjust Z until it touches.\n")
    input("  Press ENTER to begin...")

    # Home all axes
    print("\n  Homing X Y...")
    client.send_gcode("G28 X Y\nM400", timeout=60.0)
    print("  Homing Z...")
    client.send_gcode("G28 Z\nM400", timeout=60.0)

    contact_zs: list[float] = []
    final_xys: list[tuple[float, float]] = []

    for idx, (px, py) in enumerate(probe_points):
        # Retract and travel to probe point
        client.send_gcode(
            f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0,
        )
        client.send_gcode(
            f"G0 X{px:.2f} Y{py:.2f} F{f_travel:.0f}\nM400", timeout=10.0,
        )
        # Pre-descend to starting Z (safe offset above expected contact)
        client.send_gcode(
            f"G1 Z{z_start:.3f} F{f_plunge:.0f}\nM400", timeout=10.0,
        )

        z_val, final_x, final_y = _probe_single_point(
            client, config, idx, total, px, py, z_start,
        )
        contact_zs.append(z_val)
        final_xys.append((final_x, final_y))

        # Subsequent points: start 0.25 mm above the contact just found
        z_start = z_val - _SUBSEQUENT_SAFE_MARGIN_MM

    # Retract after last point
    client.send_gcode(
        f"G0 Z{z_safe:.1f} F{f_plunge:.0f}\nM400", timeout=10.0,
    )

    # ---- Compute mesh offsets ----
    mean_z = sum(contact_zs) / len(contact_zs)
    offsets_flat = [z - mean_z for z in contact_zs]

    nx, ny = probe_count
    offsets_2d: list[list[float]] = []
    for row_idx in range(ny):
        row = offsets_flat[row_idx * nx : (row_idx + 1) * nx]
        offsets_2d.append([round(v, 6) for v in row])

    # ---- Planarity check ----
    # Use the final (possibly nudged) XY positions for the plane fit
    points_xyz = [
        (fx, fy, cz)
        for (fx, fy), cz in zip(final_xys, contact_zs)
    ]
    planarity = check_planarity(points_xyz, tolerance=planarity_tolerance)

    print(f"\n{'='*60}")
    print("  BED MESH RESULTS")
    print(f"{'='*60}")
    print(f"  Mean contact Z:  {mean_z:.3f} mm")
    print(f"  Offsets range:   [{min(offsets_flat):.3f}, "
          f"{max(offsets_flat):.3f}] mm")
    print(f"  Max plane residual: {planarity.max_residual_mm:.4f} mm")

    if planarity.is_planar:
        print(f"  Planarity:       PASS (within {planarity_tolerance} mm)")
    else:
        print(f"  Planarity:       WARNING -- max residual "
              f"{planarity.max_residual_mm:.4f} mm exceeds "
              f"{planarity_tolerance} mm tolerance")
        print("  This may indicate the glass is flexing or "
              "the surface is not rigid.")
        print("  Per-point residuals:")
        for i, ((fx, fy), res) in enumerate(
            zip(final_xys, planarity.residuals)
        ):
            flag = " <<<" if abs(res) > planarity_tolerance else ""
            print(f"    Point {i+1} ({fx:.0f}, {fy:.0f}): "
                  f"{res:+.4f} mm{flag}")

    print()

    # ---- Save to config ----
    if get_yes_no("Save mesh to machine.yaml and regenerate printer.cfg?"):
        _save_bed_mesh_to_config(
            config_path, offsets_2d, mean_z, mesh_min, mesh_max,
        )
        print(f"  Updated machine.yaml (pen_work_mm = {mean_z:.3f})")

        _save_z_contact_to_jobs(mean_z)
        print(f"  Updated jobs.yaml   (z_contact_mm = {mean_z:.3f})")

        # Regenerate printer.cfg
        from robot_control.configs.loader import load_config
        from robot_control.configs.printer_cfg import generate_printer_cfg

        updated_cfg = load_config(config_path)
        printer_cfg_path = Path.home() / "printer.cfg"
        if printer_cfg_path.exists():
            backup = printer_cfg_path.with_suffix(".cfg.bak")
            printer_cfg_path.rename(backup)
            print(f"  Backed up printer.cfg -> {backup}")
        printer_cfg_text = generate_printer_cfg(updated_cfg)
        printer_cfg_path.write_text(printer_cfg_text)
        print(f"  Wrote printer.cfg to {printer_cfg_path}")
        print()
        print("  Restart Klipper to load the new mesh profile, then use:")
        print("    BED_MESH_PROFILE LOAD=default")
        print("  (Drawing scripts do this automatically.)")
    else:
        print("  Mesh NOT saved.")

    summary: dict[str, Any] = {
        "mean_contact_z": mean_z,
        "offsets": offsets_2d,
        "planarity": "PASS" if planarity.is_planar else "WARNING",
        "max_residual": planarity.max_residual_mm,
    }
    print(format_calibration_summary(summary))
    return summary
