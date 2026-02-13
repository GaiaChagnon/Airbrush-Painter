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
from typing import Any

from robot_control.calibration import patterns
from robot_control.calibration.measurement import (
    MeasurementResult,
    binary_search_z,
    calculate_new_rotation_distance,
    calculate_steps_correction,
    calculate_tool_offset,
    format_calibration_summary,
    get_choice,
    get_float_input,
    get_yes_no,
)
from robot_control.configs.loader import MachineConfig
from robot_control.hardware.job_executor import JobExecutor
from robot_control.hardware.klipper_client import KlipperClient

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
