#!/usr/bin/env python3
"""
Test basic motion functionality.

Performs motion tests to verify axes move correctly and limits are respected.

Usage:
    python -m robot_control.scripts.test_motion
    python -m robot_control.scripts.test_motion --skip-home
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.hardware.klipper_client import KlipperClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_motion(
    socket_path: str | None = None,
    config_path: str | None = None,
    skip_home: bool = False,
) -> bool:
    """
    Run motion tests.

    Returns True if all tests pass.
    """
    print("=" * 60)
    print("  MOTION TEST")
    print("=" * 60)

    # Load config
    config = load_config(config_path)
    if socket_path is None:
        socket_path = config.connection.socket_path

    client = KlipperClient(
        socket_path,
        timeout=config.connection.timeout_s,
    )

    try:
        client.connect()
        print("[OK] Connected to Klipper")

        results: list[tuple[str, bool, str]] = []

        # Test 1: Homing
        if not skip_home:
            print("\n--- Test 1: Homing X and Y ---")
            try:
                client.send_gcode("G28 X Y\nM400")
                if client.is_homed("xy"):
                    results.append(("Homing XY", True, ""))
                    print("[OK] Homing complete")
                else:
                    results.append(("Homing XY", False, "Not homed after G28"))
                    print("[FAIL] Not homed after G28")
            except Exception as e:
                results.append(("Homing XY", False, str(e)))
                print(f"[FAIL] {e}")
        else:
            print("\n--- Skipping homing (--skip-home) ---")
            if not client.is_homed("xy"):
                print("[WARN] Machine not homed - some tests may fail")

        # Test 2: Position query after home
        print("\n--- Test 2: Position Query ---")
        try:
            pos = client.get_position()
            print(f"[OK] Position: X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}")
            results.append(("Position query", True, ""))
        except Exception as e:
            results.append(("Position query", False, str(e)))
            print(f"[FAIL] {e}")

        # Test 3: X axis move
        print("\n--- Test 3: X Axis Move (50mm) ---")
        try:
            start_pos = client.get_position()
            client.send_gcode("G0 X50 F3000\nM400")
            end_pos = client.get_position()
            x_moved = abs(end_pos.x - 50.0) < 0.1
            if x_moved:
                results.append(("X axis move", True, ""))
                print(f"[OK] X moved to {end_pos.x:.3f}")
            else:
                results.append(("X axis move", False, f"Expected X=50, got {end_pos.x}"))
                print(f"[FAIL] Expected X=50, got {end_pos.x}")
        except Exception as e:
            results.append(("X axis move", False, str(e)))
            print(f"[FAIL] {e}")

        # Test 4: Y axis move
        print("\n--- Test 4: Y Axis Move (50mm) ---")
        try:
            client.send_gcode("G0 Y50 F3000\nM400")
            end_pos = client.get_position()
            y_moved = abs(end_pos.y - 50.0) < 0.1
            if y_moved:
                results.append(("Y axis move", True, ""))
                print(f"[OK] Y moved to {end_pos.y:.3f}")
            else:
                results.append(("Y axis move", False, f"Expected Y=50, got {end_pos.y}"))
                print(f"[FAIL] Expected Y=50, got {end_pos.y}")
        except Exception as e:
            results.append(("Y axis move", False, str(e)))
            print(f"[FAIL] {e}")

        # Test 5: Z axis move
        print("\n--- Test 5: Z Axis Move ---")
        try:
            z_travel = config.z_states.travel_mm
            client.send_gcode(f"G0 Z{z_travel:.3f} F300\nM400")
            end_pos = client.get_position()
            z_moved = abs(end_pos.z - z_travel) < 0.1
            if z_moved:
                results.append(("Z axis move", True, ""))
                print(f"[OK] Z moved to {end_pos.z:.3f}")
            else:
                results.append(("Z axis move", False, f"Expected Z={z_travel}, got {end_pos.z}"))
                print(f"[FAIL] Expected Z={z_travel}, got {end_pos.z}")
        except Exception as e:
            results.append(("Z axis move", False, str(e)))
            print(f"[FAIL] {e}")

        # Test 6: Rapid diagonal move
        print("\n--- Test 6: Rapid Diagonal Move ---")
        try:
            client.send_gcode("G0 X100 Y100 F6000\nM400")
            end_pos = client.get_position()
            if abs(end_pos.x - 100) < 0.1 and abs(end_pos.y - 100) < 0.1:
                results.append(("Diagonal move", True, ""))
                print(f"[OK] Moved to ({end_pos.x:.3f}, {end_pos.y:.3f})")
            else:
                results.append(("Diagonal move", False, f"Position error"))
                print(f"[FAIL] Position error")
        except Exception as e:
            results.append(("Diagonal move", False, str(e)))
            print(f"[FAIL] {e}")

        # Test 7: Return home
        print("\n--- Test 7: Return to Home ---")
        try:
            client.send_gcode("G0 X0 Y0 F6000\nM400")
            end_pos = client.get_position()
            if abs(end_pos.x) < 0.1 and abs(end_pos.y) < 0.1:
                results.append(("Return home", True, ""))
                print(f"[OK] At home position")
            else:
                results.append(("Return home", False, f"Not at home"))
                print(f"[FAIL] Not at home")
        except Exception as e:
            results.append(("Return home", False, str(e)))
            print(f"[FAIL] {e}")

        # Summary
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, ok, _ in results if ok)
        total = len(results)
        print(f"\n  Passed: {passed}/{total}")

        for name, ok, msg in results:
            status = "[OK]  " if ok else "[FAIL]"
            print(f"  {status} {name}")
            if msg:
                print(f"         {msg}")

        return passed == total

    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test motion functionality")
    parser.add_argument(
        "--socket",
        "-s",
        type=str,
        help="Klipper socket path",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="Skip homing (machine must already be homed)",
    )
    args = parser.parse_args()

    success = test_motion(args.socket, args.config, args.skip_home)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
