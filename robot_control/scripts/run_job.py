#!/usr/bin/env python3
"""
Run Job Script.

Execute a job from Job IR operations or draw test patterns.

Usage:
    python -m robot_control.scripts.run_job --pattern square
    python -m robot_control.scripts.run_job --pattern grid --interactive
    python -m robot_control.scripts.run_job --pattern calibration-suite
    python -m robot_control.scripts.run_job --file job.yaml

Available patterns:
    square, rectangle, cross, grid, circle, diagonal
    ruler-x, ruler-y, crosshair-grid, speed-test, backlash-test
    z-touch, line-weight, corner-test, fine-detail, calibration-suite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

PATTERN_MAP = {
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
    "calibration-suite": patterns.full_calibration_suite,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a job or test pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available patterns: {', '.join(PATTERN_MAP.keys())}",
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

    # Job source
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pattern",
        "-p",
        type=str,
        choices=list(PATTERN_MAP.keys()),
        help="Test pattern to draw",
    )
    source.add_argument(
        "--file",
        "-f",
        type=str,
        help="Job file to execute (YAML format)",
    )

    # Execution mode
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Use interactive mode (stroke-by-stroke)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step mode (pause after each stroke)",
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip homing (machine must already be homed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate G-code but don't execute",
    )

    # Pattern options
    parser.add_argument(
        "--size",
        type=float,
        default=50.0,
        help="Pattern size in mm (for applicable patterns)",
    )
    parser.add_argument(
        "--feed",
        type=float,
        help="Feed rate override (mm/min)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Generate operations
    ops: list[Operation] = []

    if args.pattern:
        print(f"Generating pattern: {args.pattern}")
        pattern_fn = PATTERN_MAP[args.pattern]

        # Build kwargs based on pattern
        kwargs = {}
        if args.feed:
            kwargs["feed"] = args.feed

        # Size-based patterns
        if args.pattern in ("square", "cross", "circle", "diagonal"):
            if args.pattern == "circle":
                kwargs["diameter_mm"] = args.size
            else:
                kwargs["size_mm"] = args.size

        ops = pattern_fn(**kwargs)

    elif args.file:
        print(f"Loading job file: {args.file}")
        # TODO: Implement job file loading
        print("Error: Job file loading not yet implemented")
        sys.exit(1)

    # Add homing if needed
    if not args.no_home:
        ops = [HomeXY(), ToolUp()] + ops

    # Ensure we end with tool up
    if not isinstance(ops[-1], ToolUp):
        ops.append(ToolUp())

    print(f"Job contains {len(ops)} operations")

    # Dry run - just generate G-code
    if args.dry_run:
        from robot_control.gcode.generator import GCodeGenerator

        gen = GCodeGenerator(config)
        gcode = gen.generate(ops)
        print("\n--- Generated G-code ---")
        print(gcode)
        print("--- End G-code ---")
        return

    # Execute
    socket_path = args.socket or config.connection.socket_path
    print(f"Connecting to Klipper at {socket_path}...")

    try:
        with KlipperClient(
            socket_path,
            timeout=config.connection.timeout_s,
            reconnect_attempts=config.connection.reconnect_attempts,
        ) as client:
            print("Connected.\n")

            executor = JobExecutor(client, config)

            def progress_callback(progress):
                pct = progress.progress_percent
                print(
                    f"\rProgress: {progress.completed_strokes}/{progress.total_strokes} "
                    f"({pct:.1f}%) - {progress.elapsed_time:.1f}s",
                    end="",
                    flush=True,
                )

            executor.set_progress_callback(progress_callback)

            if args.interactive or args.step:
                print("Running in interactive mode...")
                if args.step:
                    print("Step mode: Press Enter after each stroke")

                    def step_callback():
                        input("\nPress Enter for next stroke (or Ctrl+C to cancel)...")
                        return True

                    success = executor.run_interactive(
                        ops, step_mode=True, step_callback=step_callback
                    )
                else:
                    success = executor.run_interactive(ops)

                print()  # Newline after progress
                if success:
                    print("Job completed successfully.")
                else:
                    print("Job was cancelled.")
            else:
                print("Running in file mode...")
                output_path = executor.run_file(ops)
                print(f"G-code written to: {output_path}")
                print("Print started. Use Klipper/Moonraker to monitor progress.")

    except KeyboardInterrupt:
        print("\nJob interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
