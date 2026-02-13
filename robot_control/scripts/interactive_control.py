#!/usr/bin/env python3
"""Launch the interactive terminal controller.

Provides keyboard jog, tool control, and real-time position display.

Usage::

    python -m robot_control.scripts.interactive_control
    python -m robot_control.scripts.interactive_control --socket /path/to/klippy_uds

Controls::

    Arrow keys   Jog X/Y          Page Up/Down   Jog Z
    +/-          Change jog step   H              Home X Y
    P            Select pen        A              Select airbrush
    U            Tool up           D              Tool down
    O            Canvas origin     Esc            E-STOP
    Q            Quit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.hardware.interactive import InteractiveController
from robot_control.hardware.klipper_client import KlipperClient

# Keep logging quiet so the curses TUI is clean
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--socket", "-s", type=str, help="Socket path")
    parser.add_argument("--config", "-c", type=str, help="Config path")
    args = parser.parse_args()

    config = load_config(args.config)
    socket_path = args.socket or config.connection.socket_path

    client = KlipperClient(
        socket_path=socket_path,
        timeout=config.connection.timeout_s,
        reconnect_attempts=config.connection.reconnect_attempts,
        reconnect_interval=config.connection.reconnect_interval_s,
        auto_reconnect=config.connection.auto_reconnect,
    )

    try:
        client.connect()
        controller = InteractiveController(client, config)
        controller.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
