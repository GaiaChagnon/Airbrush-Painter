"""Shared pump and Klipper communication helpers.

Low-level UDS transport for Klipper API and pump-specific move/home/
endstop operations.  Used by ``test_pumps.py``, ``pump_testbed.py``,
and any future pump automation scripts.

All geometry is in **millimeters**.  Volumes are in **ml**.
Speed in **mm/s**.
"""

from __future__ import annotations

import json
import select
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_control.configs.loader import (
        MachineConfig,
        PumpMotorConfig,
        SyringeConfig,
    )

ETX = b"\x03"

_KLIPPY_ENV = Path.home() / "klippy-env" / "bin" / "python"
_KLIPPY_PY = Path.home() / "klipper" / "klippy" / "klippy.py"
_KLIPPY_LOG = Path("/tmp/klippy.log")
PRINTER_CFG_PATH = Path.home() / "printer.cfg"

_spawned_klipper_proc: subprocess.Popen | None = None

_next_id = 1


# ---------------------------------------------------------------------------
# Low-level Klipper UDS transport
# ---------------------------------------------------------------------------


def raw_send(
    sock: socket.socket,
    method: str,
    params: dict,
    timeout: float = 10.0,
) -> dict:
    """Send one JSON-RPC request and return the matching response dict.

    Parameters
    ----------
    sock : socket.socket
        Connected AF_UNIX socket to Klipper UDS.
    method : str
        Klipper API method name (e.g. ``"gcode/script"``).
    params : dict
        Parameters for the method.
    timeout : float
        Maximum seconds to wait for the response.

    Returns
    -------
    dict
        Klipper response (contains ``"result"`` or ``"error"``).
        Empty dict on timeout.
    """
    global _next_id
    req_id = _next_id
    _next_id += 1

    payload = (
        json.dumps({"id": req_id, "method": method, "params": params}).encode()
        + ETX
    )
    sock.sendall(payload)
    sock.settimeout(timeout)

    buf = b""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        while ETX not in buf and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                break

        while ETX in buf:
            idx = buf.index(ETX)
            frame = buf[:idx]
            buf = buf[idx + 1:]
            try:
                msg = json.loads(frame.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if msg.get("id") == req_id:
                return msg

        if ETX not in buf and time.monotonic() >= deadline:
            break

    return {}


def drain_socket(sock: socket.socket, duration: float = 0.2) -> None:
    """Read and discard any pending data in the socket buffer."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        try:
            data = sock.recv(4096)
            if not data:
                break
        except socket.timeout:
            break


def raw_gcode(
    sock: socket.socket,
    script: str,
    timeout: float = 30.0,
) -> bool:
    """Send G-code and return True on success."""
    resp = raw_send(sock, "gcode/script", {"script": script}, timeout)
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  !! G-code error: {msg[:120]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Klipper lifecycle
# ---------------------------------------------------------------------------


def klipper_is_alive(socket_path: str) -> bool:
    """Return True if a Klipper process owns the UDS socket."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(socket_path)
        s.close()
        return True
    except OSError:
        return False


def launch_klipper(socket_path: str) -> None:
    """Spawn Klipper as a background subprocess.

    Reads ``~/printer.cfg`` and creates the UDS socket at *socket_path*.
    Removes a stale socket file before launching.
    """
    global _spawned_klipper_proc

    sock_path = Path(socket_path)
    if sock_path.exists():
        sock_path.unlink()

    if not _KLIPPY_ENV.exists():
        raise RuntimeError(
            f"Klipper virtualenv not found at {_KLIPPY_ENV}. "
            "Cannot auto-start Klipper."
        )
    if not _KLIPPY_PY.exists():
        raise RuntimeError(
            f"klippy.py not found at {_KLIPPY_PY}. "
            "Cannot auto-start Klipper."
        )

    cmd = [
        str(_KLIPPY_ENV),
        str(_KLIPPY_PY),
        str(PRINTER_CFG_PATH),
        "-l", str(_KLIPPY_LOG),
        "-a", str(socket_path),
    ]
    print(f"  Launching Klipper: {' '.join(cmd)}")
    _spawned_klipper_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)


def restart_klipper(socket_path: str) -> None:
    """Restart Klipper, launching from scratch if not running."""
    if klipper_is_alive(socket_path):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(socket_path)
            raw_gcode(sock, "RESTART")
            sock.close()
        except OSError:
            pass
        time.sleep(3.0)
    else:
        print("  Klipper is not running -- starting it fresh ...")
        launch_klipper(socket_path)


def wait_for_ready(
    socket_path: str,
    timeout: float = 30.0,
) -> socket.socket:
    """Connect to Klipper UDS and wait until state is ``'ready'``.

    Returns
    -------
    socket.socket
        Connected socket ready for commands.

    Raises
    ------
    RuntimeError
        If Klipper doesn't become ready within *timeout*.
    """
    deadline = time.monotonic() + timeout
    restart_attempted = False

    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(socket_path)
            resp = raw_send(sock, "info", {})
            result = resp.get("result", {})
            state = result.get("state", "unknown")
            state_msg = result.get("state_message", "")

            if state == "ready":
                return sock

            if state in ("error", "shutdown") and not restart_attempted:
                restart_attempted = True
                print(f"  Klipper state: {state} -- {state_msg[:80]}")
                print("  Attempting FIRMWARE_RESTART...")
                try:
                    raw_gcode(sock, "FIRMWARE_RESTART")
                except Exception:
                    pass
                sock.close()
                time.sleep(5.0)
                continue

            sock.close()
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(1.0)

    raise RuntimeError(f"Klipper did not become ready within {timeout}s")


def emergency_disable(socket_path: str, pump_id: str) -> None:
    """Fire-and-forget motor disable for emergency shutdown."""
    try:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.settimeout(2.0)
        _sock.connect(socket_path)
        payload = (
            json.dumps({
                "id": 9999,
                "method": "gcode/script",
                "params": {
                    "script": f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=0",
                },
            }).encode()
            + ETX
        )
        _sock.sendall(payload)
        _sock.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Pump control primitives
# ---------------------------------------------------------------------------


def pump_enable(sock: socket.socket, pump_id: str) -> None:
    """Enable the pump stepper driver."""
    raw_gcode(sock, f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=1")


def pump_disable(sock: socket.socket, pump_id: str) -> None:
    """Disable the pump stepper driver."""
    raw_gcode(sock, f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=0")


def pump_set_position(sock: socket.socket, pump_id: str, pos: float) -> None:
    """Set the pump's logical position without moving."""
    raw_gcode(
        sock,
        f"MANUAL_STEPPER STEPPER={pump_id} SET_POSITION={pos:.4f}",
    )


def pump_move(
    sock: socket.socket,
    pump_id: str,
    position: float,
    speed: float,
    accel: float = 100.0,
    stop_on_endstop: int = 0,
    sync: bool = True,
    timeout: float = 60.0,
) -> bool:
    """Move the pump to an absolute position.

    Parameters
    ----------
    position : float
        Target position in mm.
    speed : float
        Travel speed in mm/s.
    accel : float
        Acceleration in mm/s^2.
    stop_on_endstop : int
        0 = ignore endstop, 1 = stop when triggered, -1 = stop when released.
    sync : bool
        If False, Klipper returns immediately (``SYNC=0``); useful for
        running multiple pumps simultaneously.
    timeout : float
        G-code response timeout in seconds.

    Returns
    -------
    bool
        True if the command was accepted by Klipper.
    """
    cmd = (
        f"MANUAL_STEPPER STEPPER={pump_id} "
        f"MOVE={position:.4f} SPEED={speed:.4f} ACCEL={accel:.1f}"
    )
    if stop_on_endstop != 0:
        cmd += f" STOP_ON_ENDSTOP={stop_on_endstop}"
    if not sync:
        cmd += " SYNC=0"
    return raw_gcode(sock, cmd, timeout=timeout)


def pump_home(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
) -> bool:
    """Home the pump to its limit switch and back off.

    Uses ``homing_direction`` to determine which way to move toward the
    switch.  After the endstop triggers, sets position to 0 and backs
    off in the opposite direction.

    Returns True on success.
    """
    travel = syringe.plunger_travel_mm
    homing_speed = pump_cfg.homing_speed_mm_s
    backoff = pump_cfg.home_backoff_mm
    h_dir = pump_cfg.homing_direction

    homing_target = h_dir * (travel + 5.0)
    dir_label = "positive" if h_dir > 0 else "negative"
    print(f"    Homing: moving {dir_label} toward limit switch "
          f"at {homing_speed} mm/s ...")
    ok = pump_move(
        sock, pump_id,
        position=homing_target,
        speed=homing_speed,
        accel=50.0,
        stop_on_endstop=1,
        timeout=60.0,
    )
    if not ok:
        print("    !! Homing move failed")
        return False

    pump_set_position(sock, pump_id, 0.0)
    print("    Position set to 0.0 mm (at limit switch)")

    if backoff > 0:
        backoff_pos = -h_dir * backoff
        print(f"    Backing off {backoff:.1f} mm ...")
        pump_move(sock, pump_id, position=backoff_pos, speed=1.0, accel=50.0)
        pump_set_position(sock, pump_id, 0.0)
        print("    Position reset to 0.0 mm (after backoff)")

    return True


def pump_home_with_backlash(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    backlash_mm: float,
) -> bool:
    """Home the pump and advance by *backlash_mm* to eliminate backlash.

    After the standard home-and-backoff sequence, advances the plunger
    by *backlash_mm* in the dispense direction, then resets position to
    0.  This takes up any lead-screw backlash so subsequent dispenses
    start from a consistent mechanical state.

    Returns True on success.
    """
    ok = pump_home(sock, pump_id, pump_cfg, syringe)
    if not ok:
        return False

    if backlash_mm > 0:
        dsign = -pump_cfg.homing_direction
        purge_pos = dsign * backlash_mm
        print(f"    Backlash purge: advancing {backlash_mm:.2f} mm ...")
        pump_move(sock, pump_id, position=purge_pos, speed=1.0, accel=50.0)
        pump_set_position(sock, pump_id, 0.0)
        print("    Position reset to 0.0 mm (backlash eliminated)")

    return True


def volume_to_mm(volume_ml: float, syringe: SyringeConfig) -> float:
    """Convert a volume in ml to plunger travel in mm."""
    return volume_ml * syringe.mm_per_ml


def mm_to_volume(mm: float, syringe: SyringeConfig) -> float:
    """Convert plunger travel in mm to volume in ml."""
    return mm * syringe.volume_per_mm


# ---------------------------------------------------------------------------
# Endstop helpers
# ---------------------------------------------------------------------------


def query_pump_endstop(
    sock: socket.socket,
    pump_id: str,
    debug: bool = False,
) -> str:
    """Query the endstop state for a manual_stepper pump.

    Returns
    -------
    str
        ``"open"``, ``"TRIGGERED"``, or ``"???"`` on failure.
    """
    drain_socket(sock)
    raw_gcode(sock, "QUERY_ENDSTOPS")
    time.sleep(0.3)
    drain_socket(sock)

    resp = raw_send(
        sock,
        "objects/query",
        {"objects": {"query_endstops": ["last_query"]}},
        timeout=3.0,
    )

    if debug:
        print(f"\n  [DEBUG] raw response: {json.dumps(resp, indent=2)}")

    status = resp.get("result", {}).get("status", {})
    last_query = status.get("query_endstops", {}).get("last_query", {})

    for key, raw_state in last_query.items():
        if pump_id in key:
            if isinstance(raw_state, int):
                return "TRIGGERED" if raw_state else "open"
            return str(raw_state)

    return "???"


def stdin_has_data() -> bool:
    """Check if stdin has data ready (non-blocking)."""
    try:
        return bool(select.select([sys.stdin], [], [], 0.0)[0])
    except Exception:
        return False
