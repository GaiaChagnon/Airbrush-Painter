#!/usr/bin/env python3
"""Run a job from test patterns or a Job IR file.

Usage::

    python -m robot_control.scripts.run_job --pattern square
    python -m robot_control.scripts.run_job --pattern grid --interactive
    python -m robot_control.scripts.run_job --pattern calibration-suite
    python -m robot_control.scripts.run_job --pattern square --step
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.calibration import patterns
from robot_control.configs.loader import load_config
from robot_control.hardware.job_executor import JobExecutor
from robot_control.hardware.klipper_client import KlipperClient
from robot_control.job_ir.operations import HomeXY, Operation, ToolUp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PATTERN_MAP: dict[str, callable] = {
    "square": patterns.square,
    "rectangle": patterns.rectangle,
    "cross": patterns.cross,
    "grid": patterns.grid,
    "circle": patterns.circle,
    "diagonal": patterns.diagonal,
    "ruler-x": patterns.ruler_x,
    "ruler-y": patterns.ruler_y,
    "crosshair-grid": patterns.crosshair_grid,
    "speed-test": patterns.speed_test,
    "backlash-test": patterns.backlash_test,
    "z-touch": patterns.z_touch_pattern,
    "line-weight": patterns.line_weight_test,
    "corner-test": patterns.corner_test,
    "fine-detail": patterns.fine_detail_test,
    "calibration-suite": patterns.calibration_suite,
    "acceleration-test": patterns.acceleration_test,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run job patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pattern", "-p",
        choices=list(PATTERN_MAP.keys()),
        required=True,
        help="Pattern to draw",
    )
    parser.add_argument("--socket", "-s", type=str, help="Socket path")
    parser.add_argument("--config", "-c", type=str, help="Config path")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Use interactive (stroke-by-stroke) mode",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step mode (pause between each stroke)",
    )
    parser.add_argument(
        "--tool", "-t",
        choices=["pen", "airbrush"],
        default="pen",
        help="Tool to use",
    )
    parser.add_argument(
        "--feed", "-f",
        type=float,
        default=None,
        help="Drawing speed override (mm/s)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    socket_path = args.socket or config.connection.socket_path

    # Build operations
    pattern_fn = PATTERN_MAP[args.pattern]
    try:
        ops: list[Operation] = pattern_fn(tool=args.tool, feed=args.feed)
    except TypeError:
        # Some patterns don't accept tool/feed kwargs
        ops = pattern_fn()

    # Prepend homing
    full_ops: list[Operation] = [HomeXY()] + ops + [ToolUp()]

    client = KlipperClient(
        socket_path=socket_path,
        timeout=config.connection.timeout_s,
        reconnect_attempts=config.connection.reconnect_attempts,
        reconnect_interval=config.connection.reconnect_interval_s,
        auto_reconnect=config.connection.auto_reconnect,
    )

    try:
        client.connect()

        # Load bed mesh profile for dynamic Z compensation (no-op if absent)
        try:
            client.send_gcode("BED_MESH_PROFILE LOAD=default")
            logger.info("Bed mesh profile loaded")
        except Exception:
            logger.debug("No bed mesh profile available (expected if uncalibrated)")

        executor = JobExecutor(client, config)

        if args.interactive or args.step:
            print(f"Running '{args.pattern}' interactively "
                  f"(step={'on' if args.step else 'off'})...")
            executor.run_interactive(full_ops, step_mode=args.step)
        else:
            print(f"Running '{args.pattern}' via file mode...")
            executor.run_file(full_ops, filename=f"{args.pattern}.gcode")
            # Monitor progress
            import time
            while True:
                prog = executor.get_file_progress()
                print(f"\r  {prog.message}", end="", flush=True)
                if prog.state.name == "IDLE":
                    break
                time.sleep(1.0)
            print()

        print("Job complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        logger.error("Job error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
