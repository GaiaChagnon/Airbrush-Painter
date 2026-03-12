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
        """Connect the high-level KlipperClient."""
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
            self._console.print("[yellow]Klipper is in shutdown -- attempting recovery...[/]")
            self._recover_from_shutdown()

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

        sock = wait_for_ready(self._cfg.connection.socket_path, timeout=45.0)
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
                if self._client is not None:
                    pos = self._client.get_position()
                    status = self._client.get_status()
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

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            time.sleep(1.0)
            try:
                status = self._client.get_status()
                if status.state == "ready":
                    self._console.print("[green]  Klipper recovered.[/]")
                    return
            except Exception:
                pass
        raise RuntimeError("Could not recover Klipper from shutdown within 30 s")

    def _restart_klipper(self) -> None:
        """Send RESTART and wait for ready."""
        self._console.print("  Restarting Klipper...")
        sp = self._cfg.connection.socket_path
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(sp)
            payload = json.dumps({
                "id": 1, "method": "gcode/script",
                "params": {"script": "RESTART"},
            }).encode() + ETX
            sock.sendall(payload)
            sock.close()
        except OSError:
            pass
        time.sleep(3.0)
        self._wait_for_klipper_ready(timeout=60.0)
        self._console.print("  [green]Klipper ready.[/]")

    def _wait_for_klipper_ready(self, timeout: float = 60.0) -> None:
        """Poll Klipper until state is 'ready' or timeout."""
        sp = self._cfg.connection.socket_path
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(sp)
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
                    state = msg.get("result", {}).get("state", "unknown")
                    if state == "ready":
                        return
            except OSError:
                pass
            time.sleep(1.0)

        self._console.print(
            f"[yellow]  WARNING: Klipper did not become ready within {timeout}s[/]"
        )
