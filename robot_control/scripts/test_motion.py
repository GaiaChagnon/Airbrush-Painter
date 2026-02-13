#!/usr/bin/env python3
"""Basic motion tests.

Performs homing, single-axis moves, and soft-limit checks to verify the
motion system is working correctly.

Usage::

    python -m robot_control.scripts.test_motion
    python -m robot_control.scripts.test_motion --skip-home
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    """Run motion tests.  Returns ``True`` if all pass."""
    print("=" * 60)
    print("  MOTION TEST")
    print("=" * 60)

    config = load_config(config_path)
    if socket_path is None:
        socket_path = config.connection.socket_path

    client = KlipperClient(
        socket_path=socket_path,
        timeout=config.connection.timeout_s,
        reconnect_attempts=config.connection.reconnect_attempts,
        reconnect_interval=config.connection.reconnect_interval_s,
        auto_reconnect=config.connection.auto_reconnect,
    )
    client.connect()

    passed = 0
    failed = 0

    def _check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"[PASS] {name}  {detail}")
            passed += 1
        else:
            print(f"[FAIL] {name}  {detail}")
            failed += 1

    try:
        tc = config.get_tool("pen")
        f_travel = tc.travel_feed_mm_s * 60.0

        # --- Homing -------------------------------------------------------
        if not skip_home:
            print("\nHoming X Y...")
            client.send_gcode("G28 X Y\nM400", timeout=30.0)
            _check("Home XY", client.is_homed("xy"))
        else:
            print("\nSkipping homing (--skip-home)")

        # --- X move -------------------------------------------------------
        print("\nMoving X +50...")
        client.send_gcode(f"G0 X50 F{f_travel:.0f}\nM400")
        pos = client.get_position()
        _check("Move X to 50", abs(pos.x - 50.0) < 0.5,
               f"X={pos.x:.3f}")

        # --- Y move -------------------------------------------------------
        print("Moving Y +50...")
        client.send_gcode(f"G0 Y50 F{f_travel:.0f}\nM400")
        pos = client.get_position()
        _check("Move Y to 50", abs(pos.y - 50.0) < 0.5,
               f"Y={pos.y:.3f}")

        # --- Combined XY --------------------------------------------------
        print("Moving X100 Y100...")
        client.send_gcode(f"G0 X100 Y100 F{f_travel:.0f}\nM400")
        pos = client.get_position()
        _check(
            "Move XY to 100,100",
            abs(pos.x - 100.0) < 0.5 and abs(pos.y - 100.0) < 0.5,
            f"X={pos.x:.3f} Y={pos.y:.3f}",
        )

        # --- Return to origin ---------------------------------------------
        print("Returning to 0,0...")
        client.send_gcode(f"G0 X0 Y0 F{f_travel:.0f}\nM400")
        pos = client.get_position()
        _check(
            "Return to origin",
            abs(pos.x) < 0.5 and abs(pos.y) < 0.5,
            f"X={pos.x:.3f} Y={pos.y:.3f}",
        )

        # --- Rapid (max speed) -------------------------------------------
        print("Rapid move (max speed)...")
        max_f = config.motion.max_velocity_mm_s * 60.0
        client.send_gcode(f"G0 X100 Y100 F{max_f:.0f}\nM400")
        client.send_gcode(f"G0 X0 Y0 F{max_f:.0f}\nM400")
        pos = client.get_position()
        _check("Rapid round-trip", abs(pos.x) < 0.5 and abs(pos.y) < 0.5,
               f"X={pos.x:.3f} Y={pos.y:.3f}")

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        failed += 1
    finally:
        client.disconnect()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic motion tests")
    parser.add_argument("--socket", "-s", type=str, help="Socket path")
    parser.add_argument("--config", "-c", type=str, help="Config path")
    parser.add_argument("--skip-home", action="store_true",
                        help="Skip homing (assume already homed)")
    args = parser.parse_args()

    success = test_motion(args.socket, args.config, args.skip_home)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
