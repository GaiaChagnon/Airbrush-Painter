#!/usr/bin/env python3
"""
Test Klipper API connection.

Verifies that the Klipper API socket is accessible and responds correctly.

Usage:
    python -m robot_control.scripts.test_connection
    python -m robot_control.scripts.test_connection --socket /path/to/klippy_uds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def test_connection(socket_path: str | None = None, config_path: str | None = None) -> bool:
    """
    Test Klipper API connection.

    Returns True if all tests pass.
    """
    print("=" * 60)
    print("  KLIPPER CONNECTION TEST")
    print("=" * 60)

    # Load config
    try:
        config = load_config(config_path)
        if socket_path is None:
            socket_path = config.connection.socket_path
        print(f"\n[OK] Configuration loaded")
    except Exception as e:
        print(f"\n[FAIL] Configuration error: {e}")
        return False

    # Check socket exists
    socket_file = Path(socket_path)
    if not socket_file.exists():
        print(f"\n[FAIL] Socket not found: {socket_path}")
        print("       Is Klipper running with -a flag?")
        print("       Example: klippy printer.cfg -a /tmp/klippy_uds")
        return False
    print(f"[OK] Socket exists: {socket_path}")

    # Test connection
    try:
        client = KlipperClient(
            socket_path,
            timeout=config.connection.timeout_s,
            reconnect_attempts=1,
        )
        client.connect()
        print("[OK] Connected to Klipper API")
    except Exception as e:
        print(f"\n[FAIL] Connection failed: {e}")
        return False

    try:
        # Test status query
        status = client.get_status()
        print(f"[OK] Printer state: {status.state}")
        print(f"     State message: {status.state_message or '(none)'}")
        print(f"     Homed axes: {status.homed_axes or '(none)'}")

        # Test position query
        pos = client.get_position()
        print(f"[OK] Position: X={pos.x:.3f} Y={pos.y:.3f} Z={pos.z:.3f}")

        # Test idle check
        idle = client.is_idle()
        print(f"[OK] Toolhead idle: {idle}")

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
        return True

    except KlipperError as e:
        print(f"\n[FAIL] Klipper error: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        return False
    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Klipper API connection")
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
    args = parser.parse_args()

    success = test_connection(args.socket, args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
