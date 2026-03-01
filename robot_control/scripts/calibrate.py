#!/usr/bin/env python3
"""Calibration entry point -- all routines are opt-in.

Usage::

    python robot_control/scripts/calibrate.py --steps-x       # X axis only
    python robot_control/scripts/calibrate.py --steps-y       # Y axis only
    python robot_control/scripts/calibrate.py --z-heights     # Z seesaw
    python robot_control/scripts/calibrate.py --tool-offset   # Tool offset
    python robot_control/scripts/calibrate.py --speed         # Speed calibration
    python robot_control/scripts/calibrate.py --endstops      # Multi-cycle endstop test
    python robot_control/scripts/calibrate.py --endstops --cycles 20
    python robot_control/scripts/calibrate.py --endstop-phase # Klipper endstop phase cal
    python robot_control/scripts/calibrate.py --servo         # Servo exercise (~30 s)
    python robot_control/scripts/calibrate.py --bed-mesh      # Surface leveling
    python robot_control/scripts/calibrate.py --bed-mesh --full-canvas
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
        description="Robot calibration (all routines are opt-in)",
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
                        help="Multi-cycle endstop repeatability test")
    parser.add_argument("--cycles", type=int, default=10,
                        help="Number of home cycles for --endstops "
                        "(default: 10, recommended: 10-20)")
    parser.add_argument("--endstop-phase", action="store_true",
                        help="Run Klipper ENDSTOP_PHASE_CALIBRATE "
                        "(requires endstop_phase.enabled in config)")
    parser.add_argument("--servo", action="store_true",
                        help="Exercise servo through full range (~30 s)")
    parser.add_argument("--bed-mesh", action="store_true",
                        help="Calibrate bed mesh (surface leveling)")
    parser.add_argument("--full-canvas", action="store_true",
                        help="Probe the full canvas area instead of "
                        "paper bounds (only with --bed-mesh)")
    args = parser.parse_args()

    selected = any([
        args.steps_x, args.steps_y, args.z_heights,
        args.tool_offset, args.speed, args.endstops,
        args.endstop_phase, args.servo, args.bed_mesh,
    ])
    if not selected:
        parser.print_help()
        print("\nError: select at least one routine (e.g. --endstops).")
        sys.exit(1)

    config = load_config(args.config)
    socket_path = args.socket or config.connection.socket_path

    # Regenerate printer.cfg from machine.yaml so Klipper limits stay in
    # sync with the config the Python code reads.
    printer_cfg_path = Path.home() / "printer.cfg"
    new_cfg_text = generate_printer_cfg(config)
    old_cfg_text = (
        printer_cfg_path.read_text() if printer_cfg_path.exists() else ""
    )
    cfg_changed = new_cfg_text != old_cfg_text
    if cfg_changed:
        if printer_cfg_path.exists():
            backup = printer_cfg_path.with_suffix(".cfg.bak")
            printer_cfg_path.rename(backup)
            print(f"Backed up old printer.cfg -> {backup.name}")
        printer_cfg_path.write_text(new_cfg_text)
        print("Regenerated printer.cfg from machine.yaml (limits synced)")

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

        # Extend idle timeout so Klipper doesn't run M84 (which clears
        # the homed state) while the user is at an interactive prompt.
        client.send_gcode(
            "SET_IDLE_TIMEOUT TIMEOUT=3600", timeout=5.0,
        )

        # Clear any previously loaded bed mesh so its compensation
        # doesn't corrupt Z probing during calibration.
        try:
            client.send_gcode("BED_MESH_CLEAR", timeout=5.0)
        except Exception:
            pass  # OK if no mesh was loaded

        # Home all axes before any calibration (Klipper rejects moves
        # unless every axis in the kinematic chain has been homed).
        print("\nHoming all axes before calibration...")
        client.send_gcode("G28\nM400", timeout=60.0)

        if args.steps_x:
            routines.calibrate_steps_per_mm(client, config, axis="X")

        if args.steps_y:
            routines.calibrate_steps_per_mm(client, config, axis="Y")

        if args.z_heights:
            routines.calibrate_z_heights(client, config)

        if args.tool_offset:
            routines.calibrate_tool_offset(client, config)

        if args.speed:
            routines.calibrate_speed(client, config)

        if args.endstops:
            routines.verify_endstops(client, config, cycles=args.cycles)

        if args.endstop_phase:
            routines.calibrate_endstop_phase(client, config)

        if args.servo:
            routines.test_servo(client, config)

        if args.bed_mesh:
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
            # Restore default idle timeout, then disable steppers
            client.send_gcode(
                f"SET_IDLE_TIMEOUT TIMEOUT="
                f"{config.motion.idle_timeout_s:.0f}",
                timeout=5.0,
            )
            client.send_gcode("M84", timeout=5.0)
        except Exception:
            pass
        client.disconnect()


if __name__ == "__main__":
    main()
