#!/usr/bin/env python3
"""
Calibration Script.

Runs calibration routines for the robot.

Usage:
    python -m robot_control.scripts.calibrate              # Full calibration
    python -m robot_control.scripts.calibrate --steps-x    # X axis only
    python -m robot_control.scripts.calibrate --steps-y    # Y axis only
    python -m robot_control.scripts.calibrate --z-heights  # Z seesaw
    python -m robot_control.scripts.calibrate --tool-offset # Tool offset
    python -m robot_control.scripts.calibrate --speed      # Speed calibration
    python -m robot_control.scripts.calibrate --endstops   # Verify endstops
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot_control.calibration import routines
from robot_control.configs.loader import load_config
from robot_control.hardware.klipper_client import KlipperClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robot calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    # Calibration options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full",
        action="store_true",
        help="Run full calibration sequence (default)",
    )
    group.add_argument(
        "--steps-x",
        action="store_true",
        help="Calibrate X axis steps/mm",
    )
    group.add_argument(
        "--steps-y",
        action="store_true",
        help="Calibrate Y axis steps/mm",
    )
    group.add_argument(
        "--z-heights",
        action="store_true",
        help="Calibrate Z seesaw heights",
    )
    group.add_argument(
        "--tool-offset",
        action="store_true",
        help="Calibrate tool offset",
    )
    group.add_argument(
        "--speed",
        action="store_true",
        help="Calibrate drawing speed",
    )
    group.add_argument(
        "--endstops",
        action="store_true",
        help="Verify endstop repeatability",
    )

    parser.add_argument(
        "--rotation-distance",
        type=float,
        default=40.0,
        help="Current rotation_distance for steps/mm calibration",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    socket_path = args.socket or config.connection.socket_path

    # Connect
    print(f"Connecting to Klipper at {socket_path}...")
    try:
        with KlipperClient(
            socket_path,
            timeout=config.connection.timeout_s,
            reconnect_attempts=config.connection.reconnect_attempts,
        ) as client:
            print("Connected.\n")

            # Run selected calibration
            if args.steps_x:
                routines.calibrate_steps_per_mm(
                    client, config, "X", 100.0, args.rotation_distance
                )
            elif args.steps_y:
                routines.calibrate_steps_per_mm(
                    client, config, "Y", 100.0, args.rotation_distance
                )
            elif args.z_heights:
                routines.calibrate_z_seesaw(client, config)
            elif args.tool_offset:
                routines.calibrate_tool_offset(client, config)
            elif args.speed:
                routines.calibrate_speed(client, config)
            elif args.endstops:
                routines.verify_endstops(client, config)
            else:
                # Default: full calibration
                routines.run_full_calibration(client, config)

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Calibration failed")
        sys.exit(1)

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
