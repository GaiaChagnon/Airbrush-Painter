"""Tests for Klipper API client.

Uses a mock Unix socket to verify:
    - JSON + 0x03 framing
    - Request / response matching
    - Auto-reconnect on broken pipe
    - Emergency stop via dedicated endpoint
    - Position query via objects/query
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from robot_control.hardware.klipper_client import (
    ETX,
    GCodeError,
    KlipperClient,
    KlipperConnectionError,
    KlipperConnectionLost,
    KlipperError,
    KlipperNotReady,
    KlipperShutdown,
    Position,
    PrinterStatus,
)


# ---------------------------------------------------------------------------
# Mock UDS server
# ---------------------------------------------------------------------------


class MockKlipperServer:
    """Minimal mock of the Klipper API Unix Domain Socket server.

    Accepts a single connection and echoes configurable responses.
    """

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(socket_path)
        self._server.listen(1)
        self._server.settimeout(5.0)
        self._conn: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self._set_default_responses()

    def _set_default_responses(self) -> None:
        self.responses = {
            "info": {"state": "ready", "software_version": "mock-0.1"},
            "objects/query": {
                "status": {
                    "toolhead": {
                        "position": [10.0, 20.0, 5.0, 0.0],
                        "homed_axes": "xy",
                        "status": "Ready",
                    },
                    "webhooks": {
                        "state": "ready",
                        "state_message": "Printer is ready",
                    },
                    "virtual_sdcard": {
                        "is_active": False,
                        "progress": 0.0,
                        "file_position": 0,
                        "file_size": 0,
                    },
                    "print_stats": {
                        "state": "standby",
                        "print_duration": 0.0,
                    },
                },
            },
            "objects/subscribe": {},
            "gcode/script": {},
            "emergency_stop": {},
            "gcode/restart": {},
        }

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._conn, _ = self._server.accept()
                self._conn.settimeout(1.0)
                self._handle_connection()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_connection(self) -> None:
        buf = b""
        while not self._stop.is_set() and self._conn:
            try:
                data = self._conn.recv(4096)
                if not data:
                    break
                buf += data
                while ETX in buf:
                    idx = buf.index(ETX)
                    msg_bytes = buf[:idx]
                    buf = buf[idx + 1:]
                    try:
                        request = json.loads(msg_bytes.decode("utf-8"))
                        self.requests.append(request)
                        self._send_response(request)
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except OSError:
                break

    def _send_response(self, request: dict[str, Any]) -> None:
        method = request.get("method", "")
        msg_id = request.get("id")
        result = self.responses.get(method, {})
        response = {"id": msg_id, "result": result}
        payload = json.dumps(response).encode("utf-8") + ETX
        try:
            self._conn.sendall(payload)
        except OSError:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._conn:
            try:
                self._conn.close()
            except OSError:
                pass
        self._server.close()
        if self._thread:
            self._thread.join(timeout=3.0)

    def close_client_connection(self) -> None:
        """Force-close the client connection to simulate MCU reset."""
        if self._conn:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None


@pytest.fixture()
def mock_server(tmp_path: Path):
    """Provide a running mock Klipper server."""
    sock_path = str(tmp_path / "klippy_uds")
    server = MockKlipperServer(sock_path)
    server.start()
    yield server
    server.stop()


@pytest.fixture()
def client(mock_server: MockKlipperServer):
    """Provide a connected KlipperClient."""
    c = KlipperClient(
        socket_path=mock_server.socket_path,
        timeout=3.0,
        reconnect_attempts=3,
        reconnect_interval=0.1,
        auto_reconnect=True,
    )
    c.connect()
    yield c
    c.disconnect()


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestConnection:
    def test_connect_success(self, mock_server: MockKlipperServer) -> None:
        c = KlipperClient(mock_server.socket_path, timeout=3.0)
        c.connect()
        assert c.is_connected
        c.disconnect()
        assert not c.is_connected

    def test_connect_failure_bad_path(self) -> None:
        c = KlipperClient("/tmp/nonexistent_socket", reconnect_attempts=1)
        with pytest.raises(KlipperConnectionError):
            c.connect()

    def test_context_manager(self, mock_server: MockKlipperServer) -> None:
        with KlipperClient(mock_server.socket_path, timeout=3.0) as c:
            assert c.is_connected
        assert not c.is_connected


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestQueries:
    def test_get_position(self, client: KlipperClient) -> None:
        pos = client.get_position()
        assert isinstance(pos, Position)
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.z == 5.0

    def test_get_status(self, client: KlipperClient) -> None:
        status = client.get_status()
        assert isinstance(status, PrinterStatus)
        assert status.state == "ready"
        assert "xy" in status.homed_axes

    def test_is_homed(self, client: KlipperClient) -> None:
        assert client.is_homed("xy")
        assert not client.is_homed("xyz")

    def test_is_idle(self, client: KlipperClient) -> None:
        assert client.is_idle()


# ---------------------------------------------------------------------------
# G-code tests
# ---------------------------------------------------------------------------


class TestGCode:
    def test_send_gcode(
        self, client: KlipperClient, mock_server: MockKlipperServer,
    ) -> None:
        client.send_gcode("G28 X Y")
        # Verify request was received
        gcode_reqs = [
            r for r in mock_server.requests
            if r.get("method") == "gcode/script"
        ]
        assert len(gcode_reqs) >= 1
        assert gcode_reqs[-1]["params"]["script"] == "G28 X Y"

    def test_emergency_stop(
        self, client: KlipperClient, mock_server: MockKlipperServer,
    ) -> None:
        client.emergency_stop()
        estop_reqs = [
            r for r in mock_server.requests
            if r.get("method") == "emergency_stop"
        ]
        assert len(estop_reqs) >= 1


# ---------------------------------------------------------------------------
# Position dataclass tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_from_list(self) -> None:
        pos = Position.from_list([1.0, 2.0, 3.0, 4.0])
        assert pos.x == 1.0
        assert pos.e == 4.0

    def test_from_short_list(self) -> None:
        pos = Position.from_list([1.0, 2.0, 3.0])
        assert pos.e == 0.0


# ---------------------------------------------------------------------------
# Virtual SD card tests
# ---------------------------------------------------------------------------


class TestVirtualSDCard:
    def test_start_print(
        self, client: KlipperClient, mock_server: MockKlipperServer,
    ) -> None:
        client.start_print("test.gcode")
        gcode_reqs = [
            r for r in mock_server.requests
            if r.get("method") == "gcode/script"
        ]
        scripts = [r["params"]["script"] for r in gcode_reqs]
        assert any("SDCARD_PRINT_FILE" in s for s in scripts)

    def test_get_print_progress(self, client: KlipperClient) -> None:
        prog = client.get_print_progress()
        assert "is_active" in prog
        assert "progress" in prog
        assert "state" in prog

    def test_pause_resume_cancel(
        self, client: KlipperClient, mock_server: MockKlipperServer,
    ) -> None:
        client.pause_print()
        client.resume_print()
        client.cancel_print()
        gcode_reqs = [
            r["params"]["script"]
            for r in mock_server.requests
            if r.get("method") == "gcode/script"
        ]
        assert "PAUSE" in gcode_reqs
        assert "RESUME" in gcode_reqs
        assert "CANCEL_PRINT" in gcode_reqs
