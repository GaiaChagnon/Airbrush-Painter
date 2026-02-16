#!/usr/bin/env python3
"""Launch the interactive terminal controller.

Provides keyboard jog, tool control, and real-time position display.

Usage::

    python robot_control/scripts/interactive_control.py
    python robot_control/scripts/interactive_control.py --socket /path/to/klippy_uds

Controls::

    Arrow keys   Jog X/Y          Page Up/Down   Jog Z
    +/-          Change jog step   H              Home X Y
    G            Go to position    P              Select pen
    A            Select airbrush   U              Tool up
    D            Tool down         O              Canvas origin
    Esc          E-STOP            Q              Quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.hardware.interactive import InteractiveController
from robot_control.hardware.klipper_client import (
    KlipperClient,
    KlipperShutdown,
)

# Keep logging quiet so the curses TUI is clean
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _recover_from_shutdown(client: KlipperClient) -> None:
    """Attempt FIRMWARE_RESTART when Klipper is in shutdown state.

    After the restart command, polls for up to 30 s until Klipper
    reports 'ready'.  Raises RuntimeError if recovery fails.
    """
    print("  Klipper is in shutdown state -- attempting FIRMWARE_RESTART...")

    # The socket is already open (connect() succeeded at the transport
    # level before raising KlipperShutdown), so we can send G-code.
    try:
        client._send_request(
            "gcode/script",
            {"script": "FIRMWARE_RESTART"},
            timeout=10.0,
        )
    except Exception:
        pass  # Expected to error during restart transition

    # Poll until ready
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            info = client._send_request("info", {}, timeout=5.0)
            state = info.get("state", "unknown")
            print(f"  Klipper state: {state}")
            if state == "ready":
                print("  Recovery successful -- Klipper is ready.")
                return
        except Exception:
            # Socket may have been reset; try reconnecting
            try:
                client._close_socket()
                import socket as _socket
                client._sock = _socket.socket(
                    _socket.AF_UNIX, _socket.SOCK_STREAM,
                )
                client._sock.settimeout(client.timeout)
                client._sock.connect(client.socket_path)
            except Exception:
                pass

    raise RuntimeError(
        "Could not recover Klipper from shutdown within 30 s. "
        "Check the MCU connection and power, then retry."
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
        try:
            client.connect()
        except KlipperShutdown:
            _recover_from_shutdown(client)

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
