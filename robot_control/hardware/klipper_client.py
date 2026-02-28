"""Klipper API client over Unix Domain Socket.

Handles:
    - Socket connection with JSON + 0x03 (ETX) framing
    - Request/response matching by unique message ID
    - Async subscription notifications
    - **Auto-reconnect** on MCU reset / Octopus power-cycle
    - Emergency stop via dedicated API endpoint (bypasses G-code queue)
    - Position / state queries via ``objects/query`` (no M114 parsing)

Protocol reference:
    https://www.klipper3d.org/API_Server.html

All timeouts and retry counts come from ``MachineConfig.connection``.
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

# ETX byte terminates each JSON message in the Klipper API protocol
ETX = b"\x03"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KlipperError(Exception):
    """Base exception for all Klipper client errors."""

    pass


class KlipperConnectionError(KlipperError):
    """Socket-level connection failure (initial or during operation)."""

    pass


class KlipperConnectionLost(KlipperError):
    """All reconnect attempts exhausted after MCU reset / power-cycle."""

    pass


class KlipperNotReady(KlipperError):
    """Klipper is not in *ready* state (still starting up or in error)."""

    pass


class GCodeError(KlipperError):
    """A ``gcode/script`` request returned an error."""

    pass


class KlipperShutdown(KlipperError):
    """Klipper entered *shutdown* state (firmware fault / e-stop)."""

    pass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Machine position in millimetres."""

    x: float
    y: float
    z: float
    e: float = 0.0

    @classmethod
    def from_list(cls, pos: list[float]) -> Position:
        """Create from Klipper's ``[x, y, z, e]`` list."""
        return cls(
            x=pos[0],
            y=pos[1],
            z=pos[2],
            e=pos[3] if len(pos) > 3 else 0.0,
        )


@dataclass
class PrinterStatus:
    """High-level printer status."""

    state: str  # "ready" | "startup" | "shutdown" | "error"
    state_message: str
    homed_axes: str  # e.g. "xy", "xyz", ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KlipperClient:
    """Klipper API UDS client with auto-reconnect.

    Parameters
    ----------
    socket_path : str
        Path to ``klippy_uds`` Unix Domain Socket.
    timeout : float
        Default per-request timeout in seconds.
    reconnect_attempts : int
        Max reconnection tries after a lost connection.
    reconnect_interval : float
        Seconds to wait between reconnection attempts.
    auto_reconnect : bool
        If ``True``, automatically attempt reconnection on socket errors
        during normal operation (not on initial ``connect()``).

    Examples
    --------
    >>> with KlipperClient("/tmp/klippy_uds") as client:
    ...     pos = client.get_position()
    ...     client.send_gcode("G28 X Y")
    """

    def __init__(
        self,
        socket_path: str,
        timeout: float = 5.0,
        reconnect_attempts: int = 5,
        reconnect_interval: float = 2.0,
        auto_reconnect: bool = True,
    ) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_interval = reconnect_interval
        self.auto_reconnect = auto_reconnect

        self._sock: socket.socket | None = None
        self._msg_id: int = 0
        self._lock = threading.Lock()
        self._recv_buffer = b""

        # Subscription state
        self._sub_callback: Callable[[dict[str, Any]], None] | None = None
        self._sub_objects: dict[str, list[str] | None] | None = None
        self._sub_thread: threading.Thread | None = None
        self._stop_sub = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` when the socket is open."""
        return self._sock is not None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection and verify Klipper is responsive.

        Raises
        ------
        KlipperConnectionError
            If the socket cannot be opened after retries.
        KlipperNotReady
            If Klipper reports a non-ready / error state.
        """
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                logger.info(
                    "Connecting to Klipper at %s (attempt %d/%d)",
                    self.socket_path,
                    attempt,
                    self.reconnect_attempts,
                )
                self._sock = socket.socket(
                    socket.AF_UNIX, socket.SOCK_STREAM,
                )
                self._sock.settimeout(self.timeout)
                self._sock.connect(self.socket_path)

                info = self._send_request("info", {})
                state = info.get("state", "unknown")

                if state == "shutdown":
                    raise KlipperShutdown(
                        f"Klipper is in shutdown state: {info}"
                    )
                if state == "error":
                    raise KlipperNotReady(
                        f"Klipper is in error state: {info}"
                    )

                logger.info("Connected to Klipper (state: %s)", state)
                return

            except socket.error as exc:
                logger.warning(
                    "Connection attempt %d failed: %s", attempt, exc,
                )
                self._close_socket()
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_interval)

        raise KlipperConnectionError(
            f"Failed to connect to Klipper at {self.socket_path} "
            f"after {self.reconnect_attempts} attempts"
        )

    def disconnect(self) -> None:
        """Cleanly close the connection."""
        self._stop_subscriptions()
        self._close_socket()
        logger.info("Disconnected from Klipper")

    def reconnect(self) -> None:
        """Re-establish a lost connection (MCU reset / power-cycle).

        Sequence:
            1. Retry the socket connection up to ``reconnect_attempts``.
            2. Send ``FIRMWARE_RESTART`` to re-initialise the MCU link.
            3. Poll ``info`` until Klipper reaches *ready* state.
            4. Re-subscribe to any active object subscriptions.

        Raises
        ------
        KlipperConnectionLost
            If all reconnection attempts are exhausted.
        """
        logger.warning("Attempting reconnection to Klipper...")
        self._close_socket()

        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                logger.info(
                    "Reconnect attempt %d/%d",
                    attempt,
                    self.reconnect_attempts,
                )
                self._sock = socket.socket(
                    socket.AF_UNIX, socket.SOCK_STREAM,
                )
                self._sock.settimeout(self.timeout)
                self._sock.connect(self.socket_path)

                # Connection is up -- try FIRMWARE_RESTART
                try:
                    self._send_request(
                        "gcode/script",
                        {"script": "FIRMWARE_RESTART"},
                        timeout=10.0,
                    )
                except KlipperError:
                    pass  # May error during restart; that's fine

                # Wait for Klipper to become ready
                if self._wait_for_ready(timeout=30.0):
                    logger.info("Reconnected successfully")
                    self._resubscribe()
                    return

            except socket.error as exc:
                logger.warning(
                    "Reconnect attempt %d failed: %s", attempt, exc,
                )
                self._close_socket()

            time.sleep(self.reconnect_interval)

        raise KlipperConnectionLost(
            f"Lost connection to Klipper at {self.socket_path} and "
            f"could not reconnect after {self.reconnect_attempts} attempts"
        )

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except socket.error:
                pass
            self._sock = None
        self._recv_buffer = b""

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON request and block until the matching response arrives.

        Interleaved subscription notifications are dispatched to the
        registered callback while waiting.

        Raises
        ------
        KlipperConnectionError
            On socket-level failure (triggers auto-reconnect if enabled).
        TimeoutError
            If the response is not received within *timeout*.
        KlipperError
            If the response contains an ``error`` field.
        """
        if not self._sock:
            raise KlipperConnectionError("Not connected to Klipper")

        msg_id = self._next_id()
        request = {"id": msg_id, "method": method, "params": params}
        payload = json.dumps(request).encode("utf-8") + ETX
        timeout = timeout or self.timeout

        with self._lock:
            try:
                self._sock.sendall(payload)
                response = self._recv_response(msg_id, timeout)
            except socket.timeout:
                raise TimeoutError(
                    f"Request '{method}' timed out after {timeout}s"
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                self._close_socket()
                if self.auto_reconnect:
                    self.reconnect()
                    # Retry the request once after successful reconnect
                    return self._send_request(
                        method, params, timeout=timeout,
                    )
                raise KlipperConnectionError(
                    f"Socket error: {exc}"
                ) from exc

        if "error" in response:
            err = response["error"]
            msg = err.get("message", str(err))
            raise KlipperError(f"Klipper error: {msg}")

        return response.get("result", {})

    def _recv_response(
        self, expected_id: int, timeout: float,
    ) -> dict[str, Any]:
        """Read messages until the one matching *expected_id* arrives."""
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Response timeout")

            self._sock.settimeout(remaining)  # type: ignore[union-attr]

            # Read until a complete message is buffered
            while ETX not in self._recv_buffer:
                chunk = self._sock.recv(4096)  # type: ignore[union-attr]
                if not chunk:
                    raise ConnectionResetError("Connection closed by server")
                self._recv_buffer += chunk

            # Extract the first complete message
            idx = self._recv_buffer.index(ETX)
            msg_bytes = self._recv_buffer[:idx]
            self._recv_buffer = self._recv_buffer[idx + 1:]

            try:
                msg = json.loads(msg_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON from Klipper: %s", exc)
                continue

            # Match by ID
            if msg.get("id") == expected_id:
                return msg

            # Dispatch subscription notification
            if "method" in msg and self._sub_callback is not None:
                try:
                    self._sub_callback(msg.get("params", {}))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Subscription callback error: %s", exc)

    # ------------------------------------------------------------------
    # Readiness polling (used during reconnect)
    # ------------------------------------------------------------------

    def _wait_for_ready(self, timeout: float = 30.0) -> bool:
        """Poll ``info`` until Klipper reports *ready* or *timeout*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                info = self._send_request("info", {}, timeout=5.0)
                if info.get("state") == "ready":
                    return True
            except (KlipperError, TimeoutError, OSError):
                pass
            time.sleep(1.0)
        return False

    # ------------------------------------------------------------------
    # G-code execution
    # ------------------------------------------------------------------

    def send_gcode(
        self,
        script: str,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Execute G-code via the Klipper API.

        Parameters
        ----------
        script : str
            One or more G-code lines (newline-separated).
        wait : bool
            If ``True``, the script should include ``M400`` so that the
            API call blocks until physical motion completes.  The API
            itself returns when the script is *accepted*, not when
            motion finishes.
        timeout : float | None
            Per-request timeout override.
        """
        try:
            self._send_request(
                "gcode/script", {"script": script}, timeout=timeout,
            )
            logger.debug(
                "Sent G-code: %s",
                script.replace("\n", " | ")[:80],
            )
        except KlipperError as exc:
            raise GCodeError(f"G-code execution failed: {exc}") from exc

    def send_gcode_with_output(
        self,
        script: str,
        *,
        timeout: float | None = None,
        collect_s: float = 3.0,
    ) -> str:
        """Execute G-code and return any console output.

        Subscribes to ``gcode/subscribe_output``, sends *script*, and
        collects ``notify_gcode_response`` notifications that arrive
        during and shortly after command execution.

        Parameters
        ----------
        script : str
            G-code or Klipper command (e.g. ``ENDSTOP_PHASE_CALIBRATE``).
        timeout : float | None
            Per-request timeout for the script itself.
        collect_s : float
            Seconds to keep reading notifications after the command
            response arrives (default 3 s).

        Returns
        -------
        str
            Concatenated console output lines.
        """
        collected: list[str] = []
        orig_callback = self._sub_callback

        def _on_notify(params: Any) -> None:
            # notify_gcode_response sends params as ["line1", "line2", ...]
            if isinstance(params, list):
                collected.extend(str(p) for p in params)
            elif isinstance(params, dict):
                resp = params.get("response")
                if resp is not None:
                    collected.append(str(resp))
            if orig_callback is not None:
                orig_callback(params)

        self._sub_callback = _on_notify
        try:
            self._send_request(
                "gcode/subscribe_output", {}, timeout=5.0,
            )
            self._send_request(
                "gcode/script", {"script": script}, timeout=timeout,
            )
            self._drain_notifications(collect_s)
        except KlipperError as exc:
            raise GCodeError(
                f"G-code execution failed: {exc}"
            ) from exc
        finally:
            self._sub_callback = orig_callback

        return "\n".join(collected)

    def _drain_notifications(self, duration_s: float) -> None:
        """Read and dispatch notifications for *duration_s* seconds.

        Processes any messages already in ``_recv_buffer`` first (they
        may have been read by ``_recv_response`` but not dispatched
        because the ID-matched response was found earlier in the
        stream), then reads the socket until the deadline.
        """
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            # Process messages already sitting in the buffer before
            # attempting any new recv -- _recv_response may have read
            # extra messages past the ID-matched response.
            while ETX in self._recv_buffer:
                idx = self._recv_buffer.index(ETX)
                raw = self._recv_buffer[:idx]
                self._recv_buffer = self._recv_buffer[idx + 1:]
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "method" in msg and self._sub_callback is not None:
                    try:
                        self._sub_callback(msg.get("params", {}))
                    except Exception:  # noqa: BLE001
                        pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._sock.settimeout(  # type: ignore[union-attr]
                min(0.5, remaining),
            )
            try:
                chunk = self._sock.recv(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                self._recv_buffer += chunk
            except socket.timeout:
                # No new data yet; loop back and check deadline
                pass

    def emergency_stop(self) -> None:
        """Immediate halt via dedicated API endpoint (not queued).

        After calling this, use ``restart()`` or ``reconnect()`` to
        recover.
        """
        logger.warning("EMERGENCY STOP triggered")
        try:
            self._send_request("emergency_stop", {})
        except KlipperShutdown:
            pass  # Expected after e-stop
        except KlipperError as exc:
            logger.error("Emergency stop error: %s", exc)

    def restart(self) -> None:
        """Restart Klipper after e-stop or error.  May need re-homing."""
        logger.info("Restarting Klipper")
        try:
            self._send_request("gcode/restart", {})
        except KlipperError:
            pass  # May error during restart transition
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_position(self) -> Position:
        """Query toolhead position via ``objects/query``.

        Returns
        -------
        Position
            Current X, Y, Z, E in mm.
        """
        result = self._send_request(
            "objects/query",
            {"objects": {"toolhead": ["position"]}},
        )
        pos = result["status"]["toolhead"]["position"]
        return Position.from_list(pos)

    def get_status(self) -> PrinterStatus:
        """Query high-level printer status.

        Returns
        -------
        PrinterStatus
            State, message, and homed axes.
        """
        result = self._send_request(
            "objects/query",
            {
                "objects": {
                    "webhooks": ["state", "state_message"],
                    "toolhead": ["homed_axes"],
                },
            },
        )
        status = result["status"]
        wh = status.get("webhooks", {})
        th = status.get("toolhead", {})
        return PrinterStatus(
            state=wh.get("state", "unknown"),
            state_message=wh.get("state_message", ""),
            homed_axes=th.get("homed_axes", ""),
        )

    def is_homed(self, axes: str = "xy") -> bool:
        """Check whether *axes* are homed (e.g. ``"xy"``, ``"xyz"``)."""
        status = self.get_status()
        return all(ax in status.homed_axes for ax in axes.lower())

    def is_idle(self) -> bool:
        """Check whether the toolhead motion queue is empty."""
        result = self._send_request(
            "objects/query",
            {"objects": {"toolhead": ["status"]}},
        )
        th_status = result["status"]["toolhead"].get("status", "Ready")
        return th_status.lower() == "ready"

    def wait_for_idle(self, timeout: float = 60.0) -> None:
        """Block until the toolhead becomes idle.

        Raises
        ------
        TimeoutError
            If still busy after *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while not self.is_idle():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Toolhead not idle after {timeout}s"
                )
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(
        self,
        objects: dict[str, list[str] | None],
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Subscribe to Klipper object-model state changes.

        Parameters
        ----------
        objects : dict
            Objects and attribute lists to watch.
            Example: ``{"toolhead": ["position"], "virtual_sdcard": None}``
        callback : callable
            Invoked with a dict of changed objects on each notification.
        """
        self._sub_callback = callback
        self._sub_objects = objects

        self._send_request("objects/subscribe", {"objects": objects})

        self._stop_sub.clear()
        self._sub_thread = threading.Thread(
            target=self._subscription_loop, daemon=True,
        )
        self._sub_thread.start()

    def unsubscribe(self) -> None:
        """Stop receiving subscription updates."""
        self._stop_subscriptions()

    def _resubscribe(self) -> None:
        """Re-establish subscriptions after a reconnect."""
        if self._sub_objects is not None and self._sub_callback is not None:
            logger.info("Re-subscribing to objects after reconnect")
            self._stop_subscriptions()
            self.subscribe(self._sub_objects, self._sub_callback)

    def _subscription_loop(self) -> None:
        """Background thread that dispatches subscription notifications."""
        while not self._stop_sub.is_set():
            try:
                if self._sock is None or self._sub_callback is None:
                    break
                self._sock.settimeout(0.1)
                try:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        # Server closed -- attempt reconnect
                        if self.auto_reconnect:
                            self.reconnect()
                        break
                    self._recv_buffer += chunk

                    while ETX in self._recv_buffer:
                        idx = self._recv_buffer.index(ETX)
                        msg_bytes = self._recv_buffer[:idx]
                        self._recv_buffer = self._recv_buffer[idx + 1:]
                        try:
                            msg = json.loads(msg_bytes.decode("utf-8"))
                            if "method" in msg:
                                self._sub_callback(
                                    msg.get("params", {}),
                                )
                        except (json.JSONDecodeError, Exception) as exc:
                            logger.warning(
                                "Subscription parse error: %s", exc,
                            )
                except socket.timeout:
                    pass
            except Exception as exc:  # noqa: BLE001
                if not self._stop_sub.is_set():
                    logger.error("Subscription loop error: %s", exc)
                break

    def _stop_subscriptions(self) -> None:
        self._stop_sub.set()
        if self._sub_thread is not None and self._sub_thread.is_alive():
            self._sub_thread.join(timeout=2.0)
        self._sub_callback = None

    # ------------------------------------------------------------------
    # Virtual SD card (file-run mode)
    # ------------------------------------------------------------------

    def start_print(self, filename: str) -> None:
        """Start printing *filename* from the ``virtual_sdcard`` dir."""
        logger.info("Starting print: %s", filename)
        self.send_gcode(f"SDCARD_PRINT_FILE FILENAME={filename}")

    def pause_print(self) -> None:
        """Pause the current file print."""
        logger.info("Pausing print")
        self.send_gcode("PAUSE")

    def resume_print(self) -> None:
        """Resume a paused file print."""
        logger.info("Resuming print")
        self.send_gcode("RESUME")

    def cancel_print(self) -> None:
        """Cancel the current file print."""
        logger.info("Cancelling print")
        self.send_gcode("CANCEL_PRINT")

    def get_print_progress(self) -> dict[str, Any]:
        """Query ``virtual_sdcard`` / ``print_stats`` progress.

        Returns
        -------
        dict
            Keys: ``is_active``, ``progress`` (0-1), ``state``,
            ``duration`` (seconds).
        """
        result = self._send_request(
            "objects/query",
            {
                "objects": {
                    "virtual_sdcard": [
                        "is_active", "progress",
                        "file_position", "file_size",
                    ],
                    "print_stats": ["state", "print_duration"],
                },
            },
        )
        s = result["status"]
        vsd = s.get("virtual_sdcard", {})
        ps = s.get("print_stats", {})
        return {
            "is_active": vsd.get("is_active", False),
            "progress": vsd.get("progress", 0.0),
            "state": ps.get("state", "unknown"),
            "duration": ps.get("print_duration", 0.0),
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> KlipperClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
