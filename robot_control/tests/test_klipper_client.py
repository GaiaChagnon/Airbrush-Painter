"""
Tests for Klipper client module.

Uses mocked sockets to test without real hardware.
"""

import json
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from robot_control.hardware.klipper_client import (
    ConnectionError,
    GCodeError,
    KlipperClient,
    KlipperError,
    KlipperNotReady,
    Position,
    PrinterStatus,
)


class TestPosition:
    """Test Position dataclass."""

    def test_from_list(self):
        """Position should be constructable from list."""
        pos = Position.from_list([10.0, 20.0, 5.0, 0.0])
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.z == 5.0
        assert pos.e == 0.0

    def test_from_list_without_e(self):
        """Position should handle 3-element list."""
        pos = Position.from_list([10.0, 20.0, 5.0])
        assert pos.x == 10.0
        assert pos.e == 0.0


class TestKlipperClientUnit:
    """Unit tests for KlipperClient without real socket."""

    def test_not_connected_initially(self):
        """Client should not be connected before connect()."""
        client = KlipperClient("/tmp/test.sock")
        assert not client.is_connected

    def test_message_id_increments(self):
        """Message IDs should increment."""
        client = KlipperClient("/tmp/test.sock")
        id1 = client._next_id()
        id2 = client._next_id()
        assert id2 == id1 + 1

    @patch("socket.socket")
    def test_connect_failure_raises(self, mock_socket):
        """Connection failure should raise ConnectionError."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = socket.error("Connection refused")
        mock_socket.return_value = mock_sock

        client = KlipperClient("/tmp/test.sock", reconnect_attempts=1)

        with pytest.raises(ConnectionError, match="Failed to connect"):
            client.connect()

    @patch("socket.socket")
    def test_disconnect_closes_socket(self, mock_socket):
        """Disconnect should close socket."""
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock

        client = KlipperClient("/tmp/test.sock")
        client._sock = mock_sock

        client.disconnect()

        mock_sock.close.assert_called_once()
        assert client._sock is None


class TestKlipperClientProtocol:
    """Test JSON protocol handling."""

    def test_etx_terminator(self):
        """Messages should be terminated with ETX (0x03)."""
        from robot_control.hardware.klipper_client import ETX

        assert ETX == b"\x03"

    def test_position_parsing(self):
        """Position data should be parsed correctly."""
        # Simulate response
        response = {
            "status": {
                "toolhead": {
                    "position": [100.5, 200.3, 10.0, 0.0],
                    "homed_axes": "xy",
                }
            }
        }

        pos_data = response["status"]["toolhead"]["position"]
        pos = Position.from_list(pos_data)

        assert pos.x == 100.5
        assert pos.y == 200.3
        assert pos.z == 10.0


class TestPrinterStatus:
    """Test PrinterStatus dataclass."""

    def test_status_fields(self):
        """PrinterStatus should store all fields."""
        status = PrinterStatus(
            state="ready",
            state_message="Printer is ready",
            homed_axes="xyz",
        )

        assert status.state == "ready"
        assert status.state_message == "Printer is ready"
        assert status.homed_axes == "xyz"


class MockKlipperServer:
    """Mock Klipper API server for integration tests."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.sock: socket.socket | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.responses: dict[str, dict] = {}

    def start(self):
        """Start mock server in background."""
        import os

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        self.sock.settimeout(1.0)
        self.running = True

        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        """Stop mock server."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()

        import os

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def set_response(self, method: str, result: dict):
        """Set response for a method."""
        self.responses[method] = result

    def _run(self):
        """Server main loop."""
        while self.running:
            try:
                conn, _ = self.sock.accept()
                self._handle_connection(conn)
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_connection(self, conn: socket.socket):
        """Handle client connection."""
        conn.settimeout(0.5)
        buffer = b""

        while self.running:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data

                while b"\x03" in buffer:
                    idx = buffer.index(b"\x03")
                    msg_bytes = buffer[:idx]
                    buffer = buffer[idx + 1 :]

                    try:
                        msg = json.loads(msg_bytes.decode())
                        response = self._handle_request(msg)
                        response_bytes = json.dumps(response).encode() + b"\x03"
                        conn.sendall(response_bytes)
                    except Exception:
                        pass

            except socket.timeout:
                continue
            except Exception:
                break

        conn.close()

    def _handle_request(self, request: dict) -> dict:
        """Generate response for request."""
        msg_id = request.get("id", 0)
        method = request.get("method", "")

        if method in self.responses:
            return {"id": msg_id, "result": self.responses[method]}

        # Default responses
        if method == "info":
            return {"id": msg_id, "result": {"state": "ready"}}
        elif method == "objects/query":
            return {
                "id": msg_id,
                "result": {
                    "status": {
                        "toolhead": {
                            "position": [0.0, 0.0, 10.0, 0.0],
                            "homed_axes": "xy",
                            "status": "Ready",
                        },
                        "webhooks": {
                            "state": "ready",
                            "state_message": "",
                        },
                    }
                },
            }
        elif method == "gcode/script":
            return {"id": msg_id, "result": {}}
        elif method == "emergency_stop":
            return {"id": msg_id, "result": {}}

        return {"id": msg_id, "error": {"message": f"Unknown method: {method}"}}


@pytest.fixture
def mock_server():
    """Create and start mock Klipper server."""
    import os
    import tempfile

    # Use short path in /tmp to avoid Unix socket path length limit
    socket_path = f"/tmp/klippy_test_{os.getpid()}.sock"
    server = MockKlipperServer(socket_path)
    server.start()
    yield server
    server.stop()


class TestKlipperClientIntegration:
    """Integration tests with mock server."""

    def test_connect_and_disconnect(self, mock_server):
        """Client should connect and disconnect cleanly."""
        client = KlipperClient(mock_server.socket_path, timeout=2.0)

        client.connect()
        assert client.is_connected

        client.disconnect()
        assert not client.is_connected

    def test_get_position(self, mock_server):
        """Client should query position successfully."""
        client = KlipperClient(mock_server.socket_path, timeout=2.0)
        client.connect()

        try:
            pos = client.get_position()
            assert isinstance(pos, Position)
            assert pos.x == 0.0
            assert pos.y == 0.0
        finally:
            client.disconnect()

    def test_get_status(self, mock_server):
        """Client should query status successfully."""
        client = KlipperClient(mock_server.socket_path, timeout=2.0)
        client.connect()

        try:
            status = client.get_status()
            assert isinstance(status, PrinterStatus)
            assert status.state == "ready"
        finally:
            client.disconnect()

    def test_send_gcode(self, mock_server):
        """Client should send G-code successfully."""
        client = KlipperClient(mock_server.socket_path, timeout=2.0)
        client.connect()

        try:
            # Should not raise
            client.send_gcode("G28 X Y")
        finally:
            client.disconnect()

    def test_is_homed(self, mock_server):
        """Client should check homing state."""
        client = KlipperClient(mock_server.socket_path, timeout=2.0)
        client.connect()

        try:
            assert client.is_homed("xy")
            assert not client.is_homed("xyz")  # z not homed in mock
        finally:
            client.disconnect()

    def test_context_manager(self, mock_server):
        """Client should work as context manager."""
        with KlipperClient(mock_server.socket_path, timeout=2.0) as client:
            assert client.is_connected
            pos = client.get_position()
            assert isinstance(pos, Position)

        assert not client.is_connected
