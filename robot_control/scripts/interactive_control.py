#!/usr/bin/env python3
"""Launch the interactive terminal controller.

Provides keyboard jog, tool control, and real-time position display.

Usage::

    python robot_control/scripts/interactive_control.py
    python robot_control/scripts/interactive_control.py --socket /path/to/klippy_uds

Controls::

    Arrow keys   Jog X/Y          Page Up/Down   Jog Z
    +/-          Change jog step   H              Home X Y Z
    G            Go to position    P              Select pen
    A            Select airbrush   U              Tool up
    D            Tool down         O              Canvas origin
    Esc          E-STOP            Q              Quit
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg
from robot_control.hardware.interactive import InteractiveController
from robot_control.hardware.klipper_client import (
    KlipperClient,
    KlipperShutdown,
)

PRINTER_CFG_PATH = Path.home() / "printer.cfg"
SOCKET_PATH = "/tmp/klippy_uds"
ETX = b"\x03"

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


def _restart_klipper_raw() -> None:
    """Send RESTART via raw socket and wait for Klipper to come back."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(SOCKET_PATH)
        payload = json.dumps({
            "id": 1, "method": "gcode/script",
            "params": {"script": "RESTART"},
        }).encode() + ETX
        sock.sendall(payload)
        sock.close()
    except OSError:
        pass
    time.sleep(3.0)


def _wait_for_klipper_ready(timeout: float = 30.0) -> None:
    """Poll Klipper until state is 'ready' or timeout."""
    deadline = time.monotonic() + timeout
    restart_attempted = False

    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            payload = json.dumps({"id": 1, "method": "info", "params": {}}).encode() + ETX
            sock.sendall(payload)
            buf = b""
            while ETX not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            sock.close()
            if ETX in buf:
                frame = buf[:buf.index(ETX)]
                msg = json.loads(frame.decode())
                result = msg.get("result", {})
                state = result.get("state", "unknown")
                if state == "ready":
                    return
                if state in ("error", "shutdown") and not restart_attempted:
                    restart_attempted = True
                    print(f"  Klipper state: {state}, attempting FIRMWARE_RESTART...")
                    _restart_klipper_raw()
                    continue
        except OSError:
            pass
        time.sleep(1.0)

    raise RuntimeError(f"Klipper did not become ready within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--socket", "-s", type=str, help="Socket path")
    parser.add_argument("--config", "-c", type=str, help="Config path")
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't regenerate printer.cfg (assume it's already correct)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    socket_path = args.socket or config.connection.socket_path

    # Regenerate printer.cfg to ensure Klipper has correct axis limits
    if not args.no_config_write:
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
        config_text = generate_printer_cfg(config)
        PRINTER_CFG_PATH.write_text(config_text)
        print(f"  Wrote printer.cfg to {PRINTER_CFG_PATH}")
        print("  Restarting Klipper...")
        _restart_klipper_raw()
        _wait_for_klipper_ready(timeout=30.0)
        print("  Klipper ready.")

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
        msg = str(exc)
        # Provide actionable advice for common Klipper errors
        if "Must home axis first" in msg:
            print("Error: Klipper requires homing before movement.")
            print("  The interactive controller homes automatically on start.")
            print("  This error may indicate a firmware or config issue.")
            print("  Try: FIRMWARE_RESTART from another terminal, then relaunch.")
        elif "Move out of range" in msg:
            print(f"Error: {msg}")
            print("  Check that machine.yaml work_area matches printer.cfg limits.")
        else:
            print(f"Error: {msg}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
