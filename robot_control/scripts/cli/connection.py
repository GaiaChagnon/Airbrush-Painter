"""Klipper connection manager for the unified CLI.

Wraps ``KlipperClient`` (for modes that use the high-level API) and a
raw ``socket.socket`` (for pump modes that talk directly to the UDS).
A background thread polls position and state so the status bar can
update without blocking the UI.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from rich.console import Console

from robot_control.configs.loader import MachineConfig, load_config
from robot_control.configs.printer_cfg import generate_printer_cfg
from robot_control.hardware.klipper_client import (
    KlipperClient,
    KlipperConnectionError,
    KlipperShutdown,
    Position,
    PrinterStatus,
)

if TYPE_CHECKING:
    from robot_control.scripts.cli.session_log import SessionLog

logger = logging.getLogger(__name__)

ETX = b"\x03"
PRINTER_CFG_PATH = Path.home() / "printer.cfg"

# Size of the socket recv buffer (bytes)
_SOCKET_RECV_BUFFER_SIZE = 4096

# Timeout for raw socket connection to Klipper (seconds)
_RAW_CONNECT_TIMEOUT_S = 45.0

# Maximum time to wait for Klipper recovery from shutdown (seconds)
_SHUTDOWN_RECOVERY_TIMEOUT_S = 30.0

# Maximum time to wait for Klipper to become ready after a restart (seconds)
_POST_RESTART_READY_TIMEOUT_S = 60.0

# Settling time after FIRMWARE_RESTART before polling again (seconds)
_FIRMWARE_RESTART_SETTLE_S = 5.0


class KlipperConnectionManager:
    """Manages Klipper connectivity for all CLI modes.

    Parameters
    ----------
    config : MachineConfig
        Machine configuration (socket path, timeouts, etc.).
    console : Rich Console for user-facing output.
    session_log : Optional session logger.
    """

    def __init__(
        self,
        config: MachineConfig,
        console: Console,
        session_log: SessionLog | None = None,
    ) -> None:
        self._cfg = config
        self._console = console
        self._session_log = session_log

        self._client: KlipperClient | None = None
        self._raw_sock: socket.socket | None = None

        self._position: Position | None = None
        self._state: str = "unknown"
        self._connected = False

        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def config(self) -> MachineConfig:
        return self._cfg

    @property
    def client(self) -> KlipperClient:
        """High-level Klipper client (connect first)."""
        if self._client is None:
            raise RuntimeError("KlipperClient not initialised -- call connect() first")
        return self._client

    @property
    def raw_socket(self) -> socket.socket:
        """Low-level UDS socket for pump modes."""
        if self._raw_sock is None:
            raise RuntimeError("Raw socket not initialised -- call connect_raw() first")
        return self._raw_sock

    def get_position(self) -> Position | None:
        with self._lock:
            return self._position

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # High-level client connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect the high-level ``KlipperClient``.

        If the initial connection attempt fails (e.g. Klipper is still
        settling after a restart), falls back to the robust
        ``_wait_for_klipper_ready`` loop -- which includes
        ``FIRMWARE_RESTART`` recovery -- before retrying.

        Raises
        ------
        KlipperConnectionError
            If Klipper is unreachable after all retries and fallback.
        """
        cfg = self._cfg
        self._client = KlipperClient(
            socket_path=cfg.connection.socket_path,
            timeout=cfg.connection.timeout_s,
            reconnect_attempts=cfg.connection.reconnect_attempts,
            reconnect_interval=cfg.connection.reconnect_interval_s,
            auto_reconnect=cfg.connection.auto_reconnect,
        )
        try:
            self._client.connect()
        except KlipperShutdown:
            self._console.print(
                "[yellow]Klipper is in shutdown"
                " -- attempting recovery...[/]"
            )
            self._recover_from_shutdown()
        except KlipperConnectionError:
            # KlipperClient exhausted its short retry window; use the
            # robust wait (with FIRMWARE_RESTART fallback) then retry.
            self._console.print(
                "[yellow]  Connection failed -- waiting for"
                " Klipper with extended timeout...[/]"
            )
            if not self._wait_for_klipper_ready(
                timeout=_POST_RESTART_READY_TIMEOUT_S,
            ):
                raise
            self._client.connect()

        with self._lock:
            self._connected = True
            self._state = "ready"
        self._start_polling()

    def disconnect(self) -> None:
        """Disconnect everything and stop polling."""
        self._stop_polling()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        if self._raw_sock is not None:
            try:
                self._raw_sock.close()
            except Exception:
                pass
            self._raw_sock = None
        with self._lock:
            self._connected = False
            self._state = "disconnected"

    # ------------------------------------------------------------------
    # Raw socket connection (for pump modes)
    # ------------------------------------------------------------------

    def connect_raw(self) -> socket.socket:
        """Open a raw UDS socket to Klipper and return it.

        Also starts background polling if not already running.
        """
        from robot_control.hardware.pump_control import wait_for_ready

        sock = wait_for_ready(self._cfg.connection.socket_path, timeout=_RAW_CONNECT_TIMEOUT_S)
        self._raw_sock = sock
        with self._lock:
            self._connected = True
            self._state = "ready"
        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._start_polling()
        return sock

    # ------------------------------------------------------------------
    # Printer config management
    # ------------------------------------------------------------------

    def regenerate_printer_cfg(self, force: bool = False) -> bool:
        """Regenerate ``printer.cfg`` from ``machine.yaml``, optionally with confirmation.

        Returns True if Klipper was restarted.
        """
        if not force:
            proceed = questionary.confirm(
                "Regenerate printer.cfg and restart Klipper?",
                default=True,
            ).ask()
            if not proceed:
                return False

        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            self._console.print(f"  Backed up printer.cfg -> {backup}")

        config_text = generate_printer_cfg(self._cfg)
        PRINTER_CFG_PATH.write_text(config_text)
        self._console.print(f"  Wrote printer.cfg to {PRINTER_CFG_PATH}")

        if self._session_log:
            self._session_log.log_action("connection", "printer_cfg_write", str(PRINTER_CFG_PATH))

        self._restart_klipper()
        return True

    def regenerate_pump_printer_cfg(self) -> None:
        """Write pump-only printer.cfg (used by pump controller mode)."""
        from robot_control.hardware.pump_control import PRINTER_CFG_PATH as PUMP_CFG_PATH

        if PUMP_CFG_PATH.exists():
            backup = PUMP_CFG_PATH.with_suffix(".cfg.bak")
            PUMP_CFG_PATH.rename(backup)

        config_text = generate_printer_cfg(self._cfg)
        PUMP_CFG_PATH.write_text(config_text)
        self._console.print(f"  Wrote printer.cfg to {PUMP_CFG_PATH}")

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def emergency_stop(self) -> None:
        """Fire E-stop through whichever connection is available."""
        if self._client is not None:
            try:
                self._client.emergency_stop()
            except Exception:
                pass
        elif self._raw_sock is not None:
            try:
                from robot_control.hardware.pump_control import raw_send
                raw_send(self._raw_sock, "emergency_stop", {}, timeout=2.0)
            except Exception:
                pass

        with self._lock:
            self._state = "shutdown"
        if self._session_log:
            self._session_log.log_action("connection", "EMERGENCY_STOP")
        logger.warning("EMERGENCY STOP triggered from CLI")

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _start_polling(self) -> None:
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="klipper-status-poll",
        )
        self._poll_thread.start()

    def _stop_polling(self) -> None:
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

    def _poll_loop(self) -> None:
        interval = self._cfg.interactive.position_poll_interval_ms / 1000.0
        while not self._poll_stop.is_set():
            try:
                with self._lock:
                    client = self._client
                if client is not None:
                    pos = client.get_position()
                    status = client.get_status()
                    with self._lock:
                        self._position = pos
                        self._state = status.state
                        self._connected = True
            except Exception:
                with self._lock:
                    self._connected = False
            self._poll_stop.wait(interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recover_from_shutdown(self) -> None:
        """Attempt FIRMWARE_RESTART after Klipper shutdown."""
        if self._client is None:
            return
        try:
            self._client.restart()
        except Exception:
            pass

        deadline = time.monotonic() + _SHUTDOWN_RECOVERY_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(1.0)
            try:
                status = self._client.get_status()
                if status.state == "ready":
                    self._console.print("[green]  Klipper recovered.[/]")
                    return
            except Exception:
                pass
        raise RuntimeError(f"Could not recover Klipper from shutdown within {_SHUTDOWN_RECOVERY_TIMEOUT_S} s")

    def _restart_klipper(self) -> None:
        """Restart Klipper, launching fresh if the process is dead.

        Uses ``pump_control.klipper_is_alive`` to detect whether
        Klipper is reachable.  If not, spawns it via
        ``pump_control.launch_klipper`` instead of sending a no-op
        RESTART to a nonexistent socket.

        Raises
        ------
        KlipperConnectionError
            If Klipper does not become ready within the timeout.
        """
        from robot_control.hardware.pump_control import (
            klipper_is_alive,
            launch_klipper,
        )

        sp = self._cfg.connection.socket_path

        if klipper_is_alive(sp):
            self._console.print("  Restarting Klipper...")
            try:
                with socket.socket(
                    socket.AF_UNIX, socket.SOCK_STREAM
                ) as sock:
                    sock.settimeout(5.0)
                    sock.connect(sp)
                    payload = json.dumps({
                        "id": 1,
                        "method": "gcode/script",
                        "params": {"script": "RESTART"},
                    }).encode() + ETX
                    sock.sendall(payload)
            except OSError:
                pass
            time.sleep(3.0)
        else:
            self._console.print(
                "  Klipper is not running -- launching fresh..."
            )
            launch_klipper(sp)

        ready = self._wait_for_klipper_ready(
            timeout=_POST_RESTART_READY_TIMEOUT_S,
        )
        if ready:
            self._console.print("  [green]Klipper ready.[/]")
        else:
            raise KlipperConnectionError(
                f"Klipper did not become ready within"
                f" {_POST_RESTART_READY_TIMEOUT_S}s after restart."
                f" Check ~/printer_data/logs/klippy.log"
            )

    def _wait_for_klipper_ready(self, timeout: float = 60.0) -> bool:
        """Poll Klipper until state is ``'ready'`` or *timeout*.

        Attempts ``FIRMWARE_RESTART`` once if Klipper reports ``error``
        or ``shutdown``, mirroring the recovery logic in
        ``pump_control.wait_for_ready``.

        Returns
        -------
        bool
            ``True`` if Klipper reached ``ready``, ``False`` on timeout.
        """
        sp = self._cfg.connection.socket_path
        deadline = time.monotonic() + timeout
        firmware_restart_attempted = False

        while time.monotonic() < deadline:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5.0)
                    sock.connect(sp)
                    payload = json.dumps(
                        {"id": 1, "method": "info", "params": {}}
                    ).encode() + ETX
                    sock.sendall(payload)
                    buf = b""
                    while ETX not in buf:
                        chunk = sock.recv(_SOCKET_RECV_BUFFER_SIZE)
                        if not chunk:
                            break
                        buf += chunk
                    if ETX in buf:
                        frame = buf[:buf.index(ETX)]
                        msg = json.loads(frame.decode())
                        result = msg.get("result", {})
                        state = result.get("state", "unknown")

                        if state == "ready":
                            return True

                        if (
                            state in ("error", "shutdown")
                            and not firmware_restart_attempted
                        ):
                            firmware_restart_attempted = True
                            state_msg = result.get(
                                "state_message", ""
                            )[:80]
                            self._console.print(
                                f"  Klipper state: {state}"
                                f" -- {state_msg}"
                            )
                            self._console.print(
                                "  Attempting FIRMWARE_RESTART..."
                            )
                            self._send_firmware_restart(sp)
                            time.sleep(_FIRMWARE_RESTART_SETTLE_S)
                            continue
            except OSError:
                pass
            time.sleep(1.0)

        self._console.print(
            f"[yellow]  WARNING: Klipper did not become ready"
            f" within {timeout}s[/]"
        )
        return False    @staticmethod
    def _send_firmware_restart(socket_path: str) -> None:
        """Send ``FIRMWARE_RESTART`` via a throwaway UDS socket."""
        try:
            with socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as sock:
                sock.settimeout(5.0)
                sock.connect(socket_path)
                payload = json.dumps({
                    "id": 1,
                    "method": "gcode/script",
                    "params": {"script": "FIRMWARE_RESTART"},
                }).encode() + ETX
                sock.sendall(payload)
        except OSError:
            pass