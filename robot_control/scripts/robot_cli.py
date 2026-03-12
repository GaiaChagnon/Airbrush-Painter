#!/usr/bin/env python3
"""Unified robot CLI -- single entry point for all control modes.

Usage::

    .venv/bin/python robot_control/scripts/robot_cli.py
    .venv/bin/python robot_control/scripts/robot_cli.py --config path/to/machine.yaml
    .venv/bin/python robot_control/scripts/robot_cli.py --no-config-write
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for all imports
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified robot control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to machine.yaml config file",
    )
    parser.add_argument(
        "--socket", "-s", type=str, default=None,
        help="Override Klipper UDS socket path",
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Skip printer.cfg regeneration on startup",
    )
    args = parser.parse_args()

    from robot_control.scripts.cli.app import RobotApp

    app = RobotApp(
        config_path=args.config,
        socket_override=args.socket,
        no_config_write=args.no_config_write,
    )
    app.run()


if __name__ == "__main__":
    main()
