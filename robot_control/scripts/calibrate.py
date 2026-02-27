#!/usr/bin/env python3
"""Calibration entry point.

Runs one or more calibration routines interactively.

Usage::

    python -m robot_control.scripts.calibrate              # Full calibration
    python -m robot_control.scripts.calibrate --steps-x    # X axis only
    python -m robot_control.scripts.calibrate --steps-y    # Y axis only
    python -m robot_control.scripts.calibrate --z-heights  # Z seesaw
    python -m robot_control.scripts.calibrate --tool-offset # Tool offset
    python -m robot_control.scripts.calibrate --speed      # Speed calibration
    python -m robot_control.scripts.calibrate --endstops   # Verify endstops
    python -m robot_control.scripts.calibrate --bed-mesh  # Surface leveling (paper bounds)
    python -m robot_control.scripts.calibrate --bed-mesh --full-canvas  # Full canvas bounds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.calibration import routines
from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg
from robot_control.hardware.klipper_client import KlipperClient, KlipperShutdown

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
    parser.add_argument("--socket", "-s", type=str, help="Socket path")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument("--steps-x", action="store_true",
                        help="Calibrate X steps/mm")
    parser.add_argument("--steps-y", action="store_true",
                        help="Calibrate Y steps/mm")
    parser.add_argument("--z-heights", action="store_true",
                        help="Calibrate Z heights")
    parser.add_argument("--tool-offset", action="store_true",
                        help="Calibrate tool XY offset")
    parser.add_argument("--speed", action="store_true",
                        help="Calibrate drawing speed")
    parser.add_argument("--endstops", action="store_true",
                        help="Verify endstop repeatability")
    parser.add_argument("--bed-mesh", action="store_true",
                        help="Calibrate bed mesh (surface leveling)")
    parser.add_argument("--full-canvas", action="store_true",
                        help="Probe the full canvas area instead of "
                        "paper bounds (only with --bed-mesh)")
    args = parser.parse_args()

    config = load_config(args.config)
    socket_path = args.socket or config.connection.socket_path

    # If no specific routine selected, run all
    run_all = not any([
        args.steps_x, args.steps_y, args.z_heights,
        args.tool_offset, args.speed, args.endstops,
        args.bed_mesh,
    ])

    # Regenerate printer.cfg from machine.yaml so Klipper limits stay in
    # sync with the config the Python code reads.
    printer_cfg_path = Path.home() / "printer.cfg"
    new_cfg_text = generate_printer_cfg(config)
    old_cfg_text = printer_cfg_path.read_text() if printer_cfg_path.exists() else ""
    cfg_changed = new_cfg_text != old_cfg_text
    if cfg_changed:
        if printer_cfg_path.exists():
            backup = printer_cfg_path.with_suffix(".cfg.bak")
            printer_cfg_path.rename(backup)
            print(f"Backed up old printer.cfg -> {backup.name}")
        printer_cfg_path.write_text(new_cfg_text)
        print(f"Regenerated printer.cfg from machine.yaml (limits synced)")

    client = KlipperClient(
        socket_path=socket_path,
        timeout=config.connection.timeout_s,
        reconnect_attempts=config.connection.reconnect_attempts,
        reconnect_interval=config.connection.reconnect_interval_s,
        auto_reconnect=config.connection.auto_reconnect,
    )

    try:
        needs_restart = cfg_changed
        try:
            client.connect()
        except KlipperShutdown:
            needs_restart = True

        if needs_restart:
            print("Restarting Klipper to apply updated printer.cfg...")
            client.reconnect()
            print("Klipper restarted successfully.")

        # Home before any calibration
        print("\nHoming X Y before calibration...")
        client.send_gcode("G28 X Y\nM400", timeout=30.0)

        if args.steps_x or run_all:
            routines.calibrate_steps_per_mm(client, config, axis="X")

        if args.steps_y or run_all:
            routines.calibrate_steps_per_mm(client, config, axis="Y")

        if args.z_heights or run_all:
            routines.calibrate_z_heights(client, config)

        if args.tool_offset or run_all:
            routines.calibrate_tool_offset(client, config)

        if args.speed or run_all:
            routines.calibrate_speed(client, config)

        if args.endstops or run_all:
            routines.verify_endstops(client, config)

        if args.bed_mesh or run_all:
            config_path = Path(args.config) if args.config else None
            routines.calibrate_bed_mesh(
                client, config,
                config_path=config_path,
                use_full_canvas=args.full_canvas,
            )

        print("\nCalibration complete.")

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    except Exception as exc:
        logger.error("Calibration error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        try:
            client.send_gcode("M84", timeout=5.0)
        except Exception:
            pass
        client.disconnect()


if __name__ == "__main__":
    main()
