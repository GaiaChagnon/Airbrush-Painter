#!/usr/bin/env python3
"""
Interactive Control Script.

Launches the terminal-based interactive controller for manual jog,
tool control, and testing.

Usage:
    python -m robot_control.scripts.interactive_control
    python -m robot_control.scripts.interactive_control --socket /path/to/klippy_uds

Controls:
    Arrow keys: Jog X/Y
    Page Up/Down: Jog Z
    +/-: Change jog increment
    H: Home X and Y
    P: Select pen tool
    A: Select airbrush tool
    U: Tool up (raise)
    D: Tool down (lower)
    O: Go to canvas origin
    Esc: Emergency stop
    Q: Quit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.hardware.interactive import InteractiveController
from robot_control.hardware.klipper_client import KlipperClient

logging.basicConfig(
    level=logging.WARNING,  # Quiet for TUI
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--socket",
        "-s",
        type=str,
        help="Klipper socket path (default: from config)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    socket_path = args.socket or config.connection.socket_path

    # Connect and run
    print(f"Connecting to Klipper at {socket_path}...")
    try:
        with KlipperClient(
            socket_path,
            timeout=config.connection.timeout_s,
            reconnect_attempts=config.connection.reconnect_attempts,
        ) as client:
            print("Connected. Entering interactive mode...")
            print("Press 'Q' to quit, 'Esc' for emergency stop.\n")

            controller = InteractiveController(client, config)
            controller.run()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("Interactive control ended.")


if __name__ == "__main__":
    main()
