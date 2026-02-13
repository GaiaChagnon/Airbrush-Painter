#!/usr/bin/env python3
"""Verify Klipper API connectivity.

Runs a quick sequence of API calls to confirm the socket is reachable,
Klipper is in a ready state, and basic queries work.

Usage::

    python -m robot_control.scripts.test_connection
    python -m robot_control.scripts.test_connection --socket /path/to/klippy_uds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow direct execution from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from robot_control.configs.loader import load_config
from robot_control.hardware.klipper_client import (
    KlipperClient,
    KlipperError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_connection(
    socket_path: str | None = None,
    config_path: str | None = None,
) -> bool:
    """Run connection tests.  Returns ``True`` if all pass."""
    print("=" * 60)
    print("  KLIPPER CONNECTION TEST")
    print("=" * 60)

    config = load_config(config_path)
    if socket_path is None:
        socket_path = config.connection.socket_path
    print(f"\n[OK] Configuration loaded")
    print(f"     Socket: {socket_path}")

    passed = 0
    failed = 0

    # --- Test 1: connect ---------------------------------------------------
    try:
        client = KlipperClient(
            socket_path=socket_path,
            timeout=config.connection.timeout_s,
            reconnect_attempts=config.connection.reconnect_attempts,
            reconnect_interval=config.connection.reconnect_interval_s,
            auto_reconnect=config.connection.auto_reconnect,
        )
        client.connect()
        print("[PASS] Socket connection established")
        passed += 1
    except Exception as exc:
        print(f"[FAIL] Connection: {exc}")
        failed += 1
        return False

    try:
        # --- Test 2: status query ------------------------------------------
        try:
            status = client.get_status()
            print(f"[PASS] Status query: state={status.state}, "
                  f"homed_axes='{status.homed_axes}'")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Status query: {exc}")
            failed += 1

        # --- Test 3: position query ----------------------------------------
        try:
            pos = client.get_position()
            print(f"[PASS] Position query: "
                  f"X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Position query: {exc}")
            failed += 1

        # --- Test 4: idle check --------------------------------------------
        try:
            idle = client.is_idle()
            print(f"[PASS] Idle check: {idle}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Idle check: {exc}")
            failed += 1

        # --- Test 5: simple G-code -----------------------------------------
        try:
            client.send_gcode("M115")  # Firmware info (harmless)
            print("[PASS] G-code send (M115)")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] G-code send: {exc}")
            failed += 1

    finally:
        client.disconnect()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Klipper connection")
    parser.add_argument("--socket", "-s", type=str, help="Socket path override")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    args = parser.parse_args()

    success = test_connection(args.socket, args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
