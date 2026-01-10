"""
Calibration Routines.

Interactive calibration procedures with measurements and calculations.
Each routine guides the user through a calibration process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

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


def calibrate_steps_per_mm(
    client: KlipperClient,
    config: MachineConfig,
    axis: str = "X",
    distance_mm: float = 100.0,
    current_rotation_distance: float = 40.0,
) -> dict[str, Any]:
    """
    Calibrate steps per mm for an axis.

    Draws a ruler pattern, prompts for measurement, calculates correction.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.
    axis : str
        Axis to calibrate ("X" or "Y").
    distance_mm : float
        Commanded distance to draw.
    current_rotation_distance : float
        Current Klipper rotation_distance value.

    Returns
    -------
    dict
        Calibration results including new rotation_distance.
    """
    print(f"\n{'=' * 50}")
    print(f"  STEPS/MM CALIBRATION - {axis} AXIS")
    print(f"{'=' * 50}")

    executor = JobExecutor(client, config)

    # Home first
    print("\n1. Homing X and Y axes...")
    executor.home_xy()
    print("   Done.")

    # Generate and draw ruler pattern
    print(f"\n2. Drawing {distance_mm}mm ruler on {axis} axis...")

    if axis.upper() == "X":
        ops = patterns.ruler_x(length_mm=distance_mm, origin=(30, 148))
    else:
        ops = patterns.ruler_y(length_mm=distance_mm, origin=(105, 50))

    executor.run_interactive(ops)
    print("   Done.")

    # Get measurement
    print(f"\n3. Measure the actual length of the ruler line.")
    print(f"   Commanded: {distance_mm}mm")
    measured = get_float_input("   Measured distance (mm)")

    # Calculate correction
    result = calculate_steps_correction(distance_mm, measured)

    print(f"\n4. Results:")
    print(f"   Error: {result.error:.3f}mm ({result.error_percent:.2f}%)")
    print(f"   Correction factor: {result.correction_factor:.6f}")

    new_rotation = calculate_new_rotation_distance(
        current_rotation_distance,
        result.correction_factor,
    )
    print(f"\n   Current rotation_distance: {current_rotation_distance}")
    print(f"   New rotation_distance:     {new_rotation:.4f}")

    # Offer to update config
    if get_yes_no("\nSave new rotation_distance to notes?"):
        _save_calibration_note(
            f"{axis}_rotation_distance",
            new_rotation,
            f"Calibrated from {current_rotation_distance}",
        )

    return {
        "axis": axis,
        "commanded": distance_mm,
        "measured": measured,
        "error_mm": result.error,
        "error_percent": result.error_percent,
        "correction_factor": result.correction_factor,
        "current_rotation_distance": current_rotation_distance,
        "new_rotation_distance": new_rotation,
    }


def calibrate_tool_offset(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """
    Calibrate XY offset between pen and airbrush.

    Draws crosshairs with both tools, prompts for offset measurement.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.

    Returns
    -------
    dict
        Calibration results with tool offset.
    """
    print(f"\n{'=' * 50}")
    print("  TOOL OFFSET CALIBRATION")
    print(f"{'=' * 50}")

    executor = JobExecutor(client, config)

    # Home first
    print("\n1. Homing X and Y axes...")
    executor.home_xy()

    # Draw crosshair with airbrush (reference tool)
    print("\n2. Drawing reference crosshair with AIRBRUSH tool...")
    print("   (Airbrush is the reference - its offset is 0,0)")

    from robot_control.job_ir.operations import SelectTool

    ops = [SelectTool(tool="airbrush")]
    ops.extend(patterns.cross(size_mm=20, center=(105, 148)))
    executor.run_interactive(ops)

    input("\n   Press Enter when ready to draw pen crosshair...")

    # Draw crosshair with pen at same commanded position
    print("\n3. Drawing crosshair with PEN tool at same position...")

    ops = [SelectTool(tool="pen")]
    ops.extend(patterns.cross(size_mm=20, center=(105, 148)))
    executor.run_interactive(ops)

    # Get measurements
    print("\n4. Measure the offset between crosshairs.")
    print("   Positive X offset means pen crosshair is to the RIGHT of airbrush")
    print("   Positive Y offset means pen crosshair is ABOVE airbrush")

    offset_x = get_float_input("   Pen X offset from airbrush (mm)", default=0.0)
    offset_y = get_float_input("   Pen Y offset from airbrush (mm)", default=0.0)

    # The offset we apply is the negative (to correct the error)
    correction_x = -offset_x
    correction_y = -offset_y

    print(f"\n5. Results:")
    print(f"   Measured offset: ({offset_x:.3f}, {offset_y:.3f})mm")
    print(f"   Correction to apply: ({correction_x:.3f}, {correction_y:.3f})mm")
    print(f"\n   Add to machine.yaml under tools.pen.xy_offset_mm:")
    print(f"   xy_offset_mm: [{correction_x:.3f}, {correction_y:.3f}]")

    return {
        "measured_offset_x": offset_x,
        "measured_offset_y": offset_y,
        "correction_x": correction_x,
        "correction_y": correction_y,
    }


def calibrate_z_seesaw(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """
    Calibrate Z heights for pen and airbrush using binary search.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.

    Returns
    -------
    dict
        Calibration results with Z heights.
    """
    print(f"\n{'=' * 50}")
    print("  Z SEESAW CALIBRATION")
    print(f"{'=' * 50}")

    from robot_control.hardware.job_executor import JobExecutor

    executor = JobExecutor(client, config)

    # Home first
    print("\n1. Homing X and Y axes...")
    executor.home_xy()

    # Move to center
    print("\n2. Moving to canvas center...")
    executor.move_to(105, 148)

    results: dict[str, Any] = {}

    # Calibrate pen Z
    print("\n3. Calibrating PEN work height...")
    print("   Place paper on the bed.")
    input("   Press Enter when ready...")

    # Start from current travel height
    current_z = config.z_states.travel_mm
    client.send_gcode(f"G0 Z{current_z:.3f} F300\nM400")

    print("\n   Moving Z to find pen touch point...")
    print("   Watch the pen tip and paper carefully.")

    pen_z = _interactive_z_search(client, config, "pen")
    if pen_z is not None:
        results["pen_work_z"] = pen_z
        print(f"\n   Pen work Z: {pen_z:.3f}mm")
    else:
        print("\n   Pen Z calibration cancelled.")

    # Calibrate airbrush Z
    if get_yes_no("\nCalibrate airbrush Z height?", default=True):
        print("\n4. Calibrating AIRBRUSH work height...")

        # Return to travel height
        client.send_gcode(f"G0 Z{config.z_states.travel_mm:.3f} F300\nM400")

        airbrush_z = _interactive_z_search(client, config, "airbrush")
        if airbrush_z is not None:
            results["airbrush_work_z"] = airbrush_z
            print(f"\n   Airbrush work Z: {airbrush_z:.3f}mm")

    # Calculate travel Z (midpoint)
    if "pen_work_z" in results and "airbrush_work_z" in results:
        travel_z = (results["pen_work_z"] + results["airbrush_work_z"]) / 2
        results["travel_z"] = travel_z
        print(f"\n   Calculated travel Z (midpoint): {travel_z:.3f}mm")

    # Return to travel height
    client.send_gcode(f"G0 Z{config.z_states.travel_mm:.3f} F300\nM400")

    print(f"\n5. Summary:")
    print(f"   Update machine.yaml z_states section:")
    for key, value in results.items():
        print(f"   {key}: {value:.3f}")

    return results


def _interactive_z_search(
    client: KlipperClient,
    config: MachineConfig,
    tool: str,
) -> float | None:
    """
    Interactive Z height search with manual control.

    Parameters
    ----------
    client : KlipperClient
        Connected client.
    config : MachineConfig
        Machine configuration.
    tool : str
        Tool being calibrated.

    Returns
    -------
    float | None
        Found Z height, or None if cancelled.
    """
    print(f"\n   Use +/- keys to adjust Z, Enter to confirm, q to cancel")
    print(f"   Increments: 1=0.1mm, 2=0.5mm, 3=1mm, 4=5mm")

    current_z = config.z_states.travel_mm
    increment = 0.5

    while True:
        print(f"\r   Z: {current_z:.3f}mm  (increment: {increment}mm)   ", end="", flush=True)

        try:
            import sys
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            if ch == "+":
                current_z += increment
            elif ch == "-":
                current_z -= increment
            elif ch == "1":
                increment = 0.1
            elif ch == "2":
                increment = 0.5
            elif ch == "3":
                increment = 1.0
            elif ch == "4":
                increment = 5.0
            elif ch in ("\r", "\n"):
                print()
                return current_z
            elif ch == "q":
                print()
                return None
            else:
                continue

            # Move Z
            client.send_gcode(f"G0 Z{current_z:.3f} F300\nM400")

        except Exception as e:
            # Fallback for non-TTY environments
            response = input(f"\n   Current Z: {current_z:.3f}. Enter new Z, +/- delta, or 'done': ")
            if response.lower() == "done":
                return current_z
            elif response.lower() in ("q", "quit"):
                return None
            elif response.startswith("+") or response.startswith("-"):
                try:
                    delta = float(response)
                    current_z += delta
                    client.send_gcode(f"G0 Z{current_z:.3f} F300\nM400")
                except ValueError:
                    print("   Invalid input")
            else:
                try:
                    current_z = float(response)
                    client.send_gcode(f"G0 Z{current_z:.3f} F300\nM400")
                except ValueError:
                    print("   Invalid input")


def calibrate_speed(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """
    Find optimal drawing speed through visual inspection.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.

    Returns
    -------
    dict
        Calibration results with recommended speed.
    """
    print(f"\n{'=' * 50}")
    print("  SPEED CALIBRATION")
    print(f"{'=' * 50}")

    executor = JobExecutor(client, config)

    # Home first
    print("\n1. Homing X and Y axes...")
    executor.home_xy()

    # Draw speed test pattern
    speeds = [500.0, 800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
    print(f"\n2. Drawing lines at speeds: {speeds} mm/min")

    ops = patterns.speed_test(
        lengths_mm=[80.0],
        speeds_mm_min=speeds,
        origin=(60, 50),
    )
    executor.run_interactive(ops)

    # Get user selection
    print("\n3. Examine the drawn lines.")
    print("   Lines are drawn from slowest (bottom) to fastest (top).")
    print("   Look for the highest speed with good line quality.")

    options = [f"{speed:.0f} mm/min" for speed in speeds]
    choice = get_choice("Select the best speed:", options)
    selected_speed = speeds[choice]

    print(f"\n4. Results:")
    print(f"   Recommended feed rate: {selected_speed:.0f} mm/min")
    print(f"\n   Update machine.yaml tools.pen.feed_mm_min:")
    print(f"   feed_mm_min: {selected_speed:.0f}")

    return {
        "tested_speeds": speeds,
        "selected_speed": selected_speed,
    }


def verify_endstops(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """
    Verify endstop repeatability.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.

    Returns
    -------
    dict
        Verification results.
    """
    print(f"\n{'=' * 50}")
    print("  ENDSTOP VERIFICATION")
    print(f"{'=' * 50}")

    results: dict[str, Any] = {"tests": [], "passed": True}

    for i in range(3):
        print(f"\n   Test {i + 1}/3: Homing...")
        client.send_gcode("G28 X Y\nM400")

        pos = client.get_position()
        results["tests"].append({"x": pos.x, "y": pos.y})

        print(f"   Position after home: X={pos.x:.4f}, Y={pos.y:.4f}")

        # Check if within tolerance
        if abs(pos.x) > 0.1 or abs(pos.y) > 0.1:
            print("   WARNING: Position not at expected 0,0")
            results["passed"] = False

        if i < 2:
            # Move away before next test
            print("   Moving away from home...")
            client.send_gcode("G0 X50 Y50 F3000\nM400")

    # Calculate repeatability
    if len(results["tests"]) >= 2:
        x_vals = [t["x"] for t in results["tests"]]
        y_vals = [t["y"] for t in results["tests"]]
        results["x_range"] = max(x_vals) - min(x_vals)
        results["y_range"] = max(y_vals) - min(y_vals)

        print(f"\n   Repeatability:")
        print(f"   X range: {results['x_range']:.4f}mm")
        print(f"   Y range: {results['y_range']:.4f}mm")

        if results["x_range"] > 0.05 or results["y_range"] > 0.05:
            print("   WARNING: Repeatability exceeds 0.05mm tolerance")
            results["passed"] = False
        else:
            print("   PASS: Repeatability within 0.05mm")

    return results


def _save_calibration_note(key: str, value: Any, note: str) -> None:
    """Save calibration note to file."""
    notes_path = Path(__file__).parent.parent / "configs" / "calibration_notes.yaml"

    notes: dict[str, Any] = {}
    if notes_path.exists():
        with open(notes_path) as f:
            notes = yaml.safe_load(f) or {}

    notes[key] = {"value": value, "note": note}

    with open(notes_path, "w") as f:
        yaml.dump(notes, f, default_flow_style=False)

    logger.info("Saved calibration note: %s = %s", key, value)


def run_full_calibration(
    client: KlipperClient,
    config: MachineConfig,
) -> dict[str, Any]:
    """
    Run complete calibration sequence.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    config : MachineConfig
        Machine configuration.

    Returns
    -------
    dict
        All calibration results.
    """
    print(f"\n{'=' * 60}")
    print("  FULL CALIBRATION SEQUENCE")
    print(f"{'=' * 60}")

    all_results: dict[str, Any] = {}

    # 1. Verify endstops
    if get_yes_no("1. Verify endstops?", default=True):
        all_results["endstops"] = verify_endstops(client, config)

    # 2. Calibrate X axis
    if get_yes_no("2. Calibrate X axis steps/mm?", default=True):
        rotation_dist = get_float_input("   Current X rotation_distance", default=40.0)
        all_results["x_axis"] = calibrate_steps_per_mm(
            client, config, "X", 100.0, rotation_dist
        )

    # 3. Calibrate Y axis
    if get_yes_no("3. Calibrate Y axis steps/mm?", default=True):
        rotation_dist = get_float_input("   Current Y rotation_distance", default=40.0)
        all_results["y_axis"] = calibrate_steps_per_mm(
            client, config, "Y", 100.0, rotation_dist
        )

    # 4. Calibrate Z heights
    if get_yes_no("4. Calibrate Z seesaw heights?", default=True):
        all_results["z_heights"] = calibrate_z_seesaw(client, config)

    # 5. Calibrate tool offset
    if "airbrush" in config.tools and "pen" in config.tools:
        if get_yes_no("5. Calibrate tool offset (pen vs airbrush)?", default=True):
            all_results["tool_offset"] = calibrate_tool_offset(client, config)

    # 6. Calibrate speed
    if get_yes_no("6. Calibrate drawing speed?", default=True):
        all_results["speed"] = calibrate_speed(client, config)

    # Print summary
    print(format_calibration_summary(all_results))

    return all_results
