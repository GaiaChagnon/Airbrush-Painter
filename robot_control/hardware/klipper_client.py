"""
Klipper API Client.

Low-level communication with Klipper via Unix Domain Socket.
Handles JSON message framing, request/response matching, and subscriptions.

Protocol:
    - Messages are JSON objects terminated by 0x03 (ETX byte)
    - Each request has a unique ID for response matching
    - Subscriptions deliver async state updates
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ETX byte terminates each JSON message
ETX = b"\x03"
ETX_CHAR = "\x03"


class KlipperError(Exception):
    """Base exception for Klipper client errors."""

    pass


class ConnectionError(KlipperError):
    """Raised when socket connection fails."""

    pass


class KlipperNotReady(KlipperError):
    """Raised when Klipper is not in ready state."""

    pass


class GCodeError(KlipperError):
    """Raised when G-code execution fails."""

    pass


class KlipperShutdown(KlipperError):
    """Raised when Klipper enters shutdown state."""

    pass


@dataclass
class Position:
    """Machine position in millimeters."""

    x: float
    y: float
    z: float
    e: float = 0.0  # Extruder position (unused for pen/airbrush)

    @classmethod
    def from_list(cls, pos: list[float]) -> Position:
        """Create Position from Klipper position list [x, y, z, e]."""
        return cls(x=pos[0], y=pos[1], z=pos[2], e=pos[3] if len(pos) > 3 else 0.0)


@dataclass
class PrinterStatus:
    """Klipper printer status."""

    state: str  # "ready", "startup", "shutdown", "error"
    state_message: str
    homed_axes: str  # e.g., "xyz", "xy", ""


class KlipperClient:
    """
    Klipper API UDS client.

    Provides structured access to Klipper's JSON API over Unix Domain Socket.
    Handles connection management, message framing, and response matching.

    Parameters
    ----------
    socket_path : str
        Path to Klipper's Unix Domain Socket (e.g., "/tmp/klippy_uds").
    timeout : float
        Request timeout in seconds. Default: 5.0.
    reconnect_attempts : int
        Number of reconnection attempts on failure. Default: 3.

    Examples
    --------
    >>> client = KlipperClient("/tmp/klippy_uds")
    >>> client.connect()
    >>> pos = client.get_position()
    >>> print(f"Position: X={pos.x}, Y={pos.y}, Z={pos.z}")
    >>> client.send_gcode("G28 X Y")
    >>> client.disconnect()
    """

    def __init__(
        self,
        socket_path: str,
        timeout: float = 5.0,
        reconnect_attempts: int = 3,
    ) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts

        self._sock: socket.socket | None = None
        self._msg_id = 0
        self._lock = threading.Lock()
        self._recv_buffer = b""

        # Subscription handling
        self._subscription_callback: Callable[[dict[str, Any]], None] | None = None
        self._subscription_thread: threading.Thread | None = None
        self._stop_subscription = threading.Event()

    @property
    def is_connected(self) -> bool:
        """Check if socket connection is active."""
        return self._sock is not None

    def connect(self) -> None:
        """
        Establish socket connection to Klipper API.

        Raises
        ------
        ConnectionError
            If connection fails after all retry attempts.
        KlipperNotReady
            If Klipper is not in ready state.
        """
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                logger.info(
                    "Connecting to Klipper at %s (attempt %d/%d)",
                    self.socket_path,
                    attempt,
                    self.reconnect_attempts,
                )

                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.settimeout(self.timeout)
                self._sock.connect(self.socket_path)

                # Verify connection with info request
                info = self._send_request("info", {})
                state = info.get("state", "unknown")

                if state == "shutdown":
                    raise KlipperShutdown(f"Klipper is in shutdown state: {info}")
                elif state == "error":
                    raise KlipperNotReady(f"Klipper is in error state: {info}")

                logger.info("Connected to Klipper (state: %s)", state)
                return

            except socket.error as e:
                logger.warning("Connection attempt %d failed: %s", attempt, e)
                self._close_socket()
                if attempt < self.reconnect_attempts:
                    time.sleep(1.0)

        raise ConnectionError(
            f"Failed to connect to Klipper at {self.socket_path} "
            f"after {self.reconnect_attempts} attempts"
        )

    def disconnect(self) -> None:
        """Close socket connection."""
        self._stop_subscriptions()
        self._close_socket()
        logger.info("Disconnected from Klipper")

    def _close_socket(self) -> None:
        """Close socket and reset state."""
        if self._sock:
            try:
                self._sock.close()
            except socket.error:
                pass
            self._sock = None
        self._recv_buffer = b""

    def _next_id(self) -> int:
        """Get next unique message ID."""
        self._msg_id += 1
        return self._msg_id

    def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Send request and wait for response.

        Parameters
        ----------
        method : str
            API method name (e.g., "info", "gcode/script").
        params : dict
            Method parameters.
        timeout : float | None
            Request timeout. Uses default if None.

        Returns
        -------
        dict
            Response result field.

        Raises
        ------
        ConnectionError
            If not connected.
        TimeoutError
            If response not received within timeout.
        KlipperError
            If response contains error.
        """
        if not self._sock:
            raise ConnectionError("Not connected to Klipper")

        msg_id = self._next_id()
        request = {"id": msg_id, "method": method, "params": params}
        request_bytes = json.dumps(request).encode("utf-8") + ETX

        timeout = timeout or self.timeout

        with self._lock:
            try:
                self._sock.sendall(request_bytes)
                response = self._recv_response(msg_id, timeout)
            except socket.timeout:
                raise TimeoutError(f"Request '{method}' timed out after {timeout}s")
            except socket.error as e:
                self._close_socket()
                raise ConnectionError(f"Socket error: {e}")

        if "error" in response:
            error = response["error"]
            msg = error.get("message", str(error))
            raise KlipperError(f"Klipper error: {msg}")

        return response.get("result", {})

    def _recv_response(self, expected_id: int, timeout: float) -> dict[str, Any]:
        """
        Receive and parse response with matching ID.

        Handles subscription notifications that may arrive before response.
        """
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Response timeout")

            self._sock.settimeout(remaining)

            # Read until we have a complete message
            while ETX not in self._recv_buffer:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by server")
                self._recv_buffer += chunk

            # Extract first complete message
            idx = self._recv_buffer.index(ETX)
            msg_bytes = self._recv_buffer[:idx]
            self._recv_buffer = self._recv_buffer[idx + 1 :]

            try:
                msg = json.loads(msg_bytes.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON response: %s", e)
                continue

            # Check if this is our response
            if msg.get("id") == expected_id:
                return msg

            # Handle subscription notification
            if "method" in msg and self._subscription_callback:
                try:
                    self._subscription_callback(msg.get("params", {}))
                except Exception as e:
                    logger.error("Subscription callback error: %s", e)

    # --- G-code Execution ---

    def send_gcode(self, script: str, *, wait: bool = False, timeout: float | None = None) -> None:
        """
        Execute G-code via Klipper API.

        Parameters
        ----------
        script : str
            G-code to execute (single line or multi-line).
        wait : bool
            If True, block until G-code execution completes.
            Note: completion means script was processed, not motion finished.
            Use M400 in the script to wait for motion completion.
        timeout : float | None
            Request timeout in seconds. If None, uses the client's default timeout.
            Some commands like STEPPER_BUZZ take longer and need increased timeout.

        Raises
        ------
        GCodeError
            If G-code execution fails.
        """
        try:
            self._send_request("gcode/script", {"script": script}, timeout=timeout)
            logger.debug("Sent G-code: %s", script.replace("\n", " | ")[:80])
        except KlipperError as e:
            raise GCodeError(f"G-code execution failed: {e}") from e

    def emergency_stop(self) -> None:
        """
        Trigger immediate emergency stop.

        Uses dedicated API endpoint that bypasses G-code queue.
        After emergency stop, use restart() to recover.
        """
        logger.warning("EMERGENCY STOP triggered")
        try:
            self._send_request("emergency_stop", {})
        except KlipperShutdown:
            pass  # Expected after emergency stop
        except KlipperError as e:
            logger.error("Emergency stop error: %s", e)

    def restart(self) -> None:
        """
        Restart Klipper after emergency stop or error.

        May need to re-home after restart.
        """
        logger.info("Restarting Klipper")
        try:
            self._send_request("gcode/restart", {})
        except KlipperError:
            pass  # May get error during restart

        # Wait for restart to complete
        time.sleep(2.0)

    # --- State Queries ---

    def get_position(self) -> Position:
        """
        Query current toolhead position.

        Returns structured Position object, not parsed text.

        Returns
        -------
        Position
            Current X, Y, Z, E position in mm.
        """
        result = self._send_request(
            "objects/query",
            {"objects": {"toolhead": ["position"]}},
        )
        pos = result["status"]["toolhead"]["position"]
        return Position.from_list(pos)

    def get_status(self) -> PrinterStatus:
        """
        Query printer state.

        Returns
        -------
        PrinterStatus
            Current state, message, and homed axes.
        """
        result = self._send_request(
            "objects/query",
            {
                "objects": {
                    "webhooks": ["state", "state_message"],
                    "toolhead": ["homed_axes"],
                }
            },
        )
        status = result["status"]
        webhooks = status.get("webhooks", {})
        toolhead = status.get("toolhead", {})

        return PrinterStatus(
            state=webhooks.get("state", "unknown"),
            state_message=webhooks.get("state_message", ""),
            homed_axes=toolhead.get("homed_axes", ""),
        )

    def is_homed(self, axes: str = "xy") -> bool:
        """
        Check if specified axes are homed.

        Parameters
        ----------
        axes : str
            Axes to check (e.g., "xy", "xyz").

        Returns
        -------
        bool
            True if all specified axes are homed.
        """
        status = self.get_status()
        return all(ax in status.homed_axes for ax in axes.lower())

    def is_idle(self) -> bool:
        """
        Check if toolhead is idle (no pending moves).

        Returns
        -------
        bool
            True if motion queue is empty.
        """
        result = self._send_request(
            "objects/query",
            {"objects": {"toolhead": ["status"]}},
        )
        status = result["status"]["toolhead"].get("status", "Ready")
        return status.lower() == "ready"

    def wait_for_idle(self, timeout: float = 60.0) -> None:
        """
        Wait until toolhead is idle.

        Parameters
        ----------
        timeout : float
            Maximum wait time in seconds.

        Raises
        ------
        TimeoutError
            If not idle within timeout.
        """
        deadline = time.monotonic() + timeout
        while not self.is_idle():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Toolhead not idle after {timeout}s")
            time.sleep(0.1)

    # --- Subscriptions ---

    def subscribe(
        self,
        objects: dict[str, list[str] | None],
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Subscribe to object state changes.

        Parameters
        ----------
        objects : dict
            Objects and attributes to subscribe to.
            Example: {"toolhead": ["position"], "virtual_sdcard": None}
        callback : callable
            Function called with state updates. Receives dict of changed objects.
        """
        self._subscription_callback = callback

        self._send_request("objects/subscribe", {"objects": objects})

        # Start background thread to handle notifications
        self._stop_subscription.clear()
        self._subscription_thread = threading.Thread(
            target=self._subscription_loop,
            daemon=True,
        )
        self._subscription_thread.start()

    def _subscription_loop(self) -> None:
        """Background thread for handling subscription updates."""
        while not self._stop_subscription.is_set():
            try:
                if self._sock and self._subscription_callback:
                    # Check for pending messages
                    self._sock.settimeout(0.1)
                    try:
                        chunk = self._sock.recv(4096)
                        if chunk:
                            self._recv_buffer += chunk

                            # Process complete messages
                            while ETX in self._recv_buffer:
                                idx = self._recv_buffer.index(ETX)
                                msg_bytes = self._recv_buffer[:idx]
                                self._recv_buffer = self._recv_buffer[idx + 1 :]

                                try:
                                    msg = json.loads(msg_bytes.decode("utf-8"))
                                    if "method" in msg:
                                        self._subscription_callback(msg.get("params", {}))
                                except (json.JSONDecodeError, Exception) as e:
                                    logger.warning("Subscription parse error: %s", e)

                    except socket.timeout:
                        pass

            except Exception as e:
                if not self._stop_subscription.is_set():
                    logger.error("Subscription loop error: %s", e)
                break

    def _stop_subscriptions(self) -> None:
        """Stop subscription handling."""
        self._stop_subscription.set()
        if self._subscription_thread and self._subscription_thread.is_alive():
            self._subscription_thread.join(timeout=1.0)
        self._subscription_callback = None

    # --- Virtual SD Card (File-Run Mode) ---

    def start_print(self, filename: str) -> None:
        """
        Start printing a file from virtual_sdcard.

        Parameters
        ----------
        filename : str
            Name of G-code file in virtual_sdcard directory.
        """
        logger.info("Starting print: %s", filename)
        self.send_gcode(f"SDCARD_PRINT_FILE FILENAME={filename}")

    def pause_print(self) -> None:
        """Pause current print."""
        logger.info("Pausing print")
        self.send_gcode("PAUSE")

    def resume_print(self) -> None:
        """Resume paused print."""
        logger.info("Resuming print")
        self.send_gcode("RESUME")

    def cancel_print(self) -> None:
        """Cancel current print."""
        logger.info("Cancelling print")
        self.send_gcode("CANCEL_PRINT")

    def get_print_progress(self) -> dict[str, Any]:
        """
        Query virtual_sdcard print progress.

        Returns
        -------
        dict
            Progress info: is_active, progress, file_position, file_size.
        """
        result = self._send_request(
            "objects/query",
            {
                "objects": {
                    "virtual_sdcard": ["is_active", "progress", "file_position", "file_size"],
                    "print_stats": ["state", "print_duration"],
                }
            },
        )
        status = result["status"]
        return {
            "is_active": status.get("virtual_sdcard", {}).get("is_active", False),
            "progress": status.get("virtual_sdcard", {}).get("progress", 0.0),
            "state": status.get("print_stats", {}).get("state", "unknown"),
            "duration": status.get("print_stats", {}).get("print_duration", 0.0),
        }

    # --- Context Manager ---

    def __enter__(self) -> KlipperClient:
        """Connect on context entry."""
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        """Disconnect on context exit."""
        self.disconnect()
