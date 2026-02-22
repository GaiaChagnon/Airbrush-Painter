#!/usr/bin/env python3
"""Syringe pump hardware bring-up and testing.

Writes a minimal ``printer.cfg`` (``kinematics: none``) with
``[manual_stepper]`` sections for each pump.  XYZ axes are not
included -- only the selected pump motor is active.

All pump parameters (pins, syringe geometry, speeds, endstop) come
from ``machine.yaml`` via the config loader.

Phases
------
    Phase 1 -- Motor spin:      basic forward/reverse movement
    Phase 2 -- Limit switch:    interactive endstop verification
    Phase 3 -- Homing:          home to limit switch + backoff
    Phase 4 -- Full travel:     0 -> plunger_travel_mm -> 0
    Phase 5 -- Continuous cycle: dispense + retract N times (user-requested)
    Phase 6 -- Precise volume:  dispense exact volume (user-requested)
    Phase 7 -- Micro-dose:      tiny volumes (0.01, 0.05, 0.1 ml)
    Phase 8 -- Speed ramp:      dispense at increasing speeds
    Phase 9 -- Repeatability:   N full cycles then re-home step-loss check

Usage::

    python robot_control/scripts/test_motors.pypython robot_control/scripts/test_motors.py
    python robot_control/scripts/test_pumps.py --test spin
    python robot_control/scripts/test_pumps.py --test cycle --cycles 10
    python robot_control/scripts/test_pumps.py --test dispense --volume 0.25
    python robot_control/scripts/test_pumps.py --pump pump_0 --speed 3
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import (
    PumpMotorConfig,
    PumpsConfig,
    SyringeConfig,
    load_config,
)
from robot_control.configs.printer_cfg import generate_pump_test_cfg

# ---------------------------------------------------------------------------
# Config -- loaded once at import time
# ---------------------------------------------------------------------------

_CFG = load_config()

SOCKET_PATH = _CFG.connection.socket_path
PRINTER_CFG_PATH = Path.home() / "printer.cfg"
ETX = b"\x03"

# ---------------------------------------------------------------------------
# Low-level Klipper comms (same pattern as test_motors.py)
# ---------------------------------------------------------------------------

_next_id = 1


def _raw_send(
    sock: socket.socket,
    method: str,
    params: dict,
    timeout: float = 10.0,
) -> dict:
    """Send one JSON request and return the matching response dict."""
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


def _drain_socket(sock: socket.socket, duration: float = 0.2) -> None:
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


def _raw_gcode(
    sock: socket.socket,
    script: str,
    timeout: float = 30.0,
) -> bool:
    """Send G-code and return True on success."""
    resp = _raw_send(sock, "gcode/script", {"script": script}, timeout)
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        print(f"  !! G-code error: {msg[:120]}")
        return False
    return True


def _wait_for_ready(timeout: float = 30.0) -> socket.socket:
    """Connect to Klipper UDS and wait until state is 'ready'."""
    deadline = time.monotonic() + timeout
    restart_attempted = False

    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            resp = _raw_send(sock, "info", {})
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
                    _raw_gcode(sock, "FIRMWARE_RESTART")
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


_KLIPPY_ENV = Path.home() / "klippy-env" / "bin" / "python"
_KLIPPY_PY = Path.home() / "klipper" / "klippy" / "klippy.py"
_KLIPPY_LOG = Path("/tmp/klippy.log")

# Track a Klipper process we spawned so we can clean up on exit.
_spawned_klipper_proc: subprocess.Popen | None = None


def _klipper_is_alive() -> bool:
    """Return True if a Klipper process owns the UDS socket."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(SOCKET_PATH)
        s.close()
        return True
    except OSError:
        return False


def _launch_klipper() -> None:
    """Spawn Klipper as a background subprocess.

    The process reads ``~/printer.cfg`` and creates the UDS socket at
    ``SOCKET_PATH``.  If a previous socket file is stale we remove it
    before launching.
    """
    global _spawned_klipper_proc

    sock_path = Path(SOCKET_PATH)
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
        "-a", str(SOCKET_PATH),
    ]
    print(f"  Launching Klipper: {' '.join(cmd)}")
    _spawned_klipper_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)


def restart_klipper() -> None:
    """Restart Klipper, launching it from scratch if not running.

    First tries to send a ``RESTART`` command over the UDS socket.
    If the socket is dead (e.g. after power loss), spawns a fresh
    Klipper process instead.
    """
    if _klipper_is_alive():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            _raw_gcode(sock, "RESTART")
            sock.close()
        except OSError:
            pass
        time.sleep(3.0)
    else:
        print("  Klipper is not running -- starting it fresh ...")
        _launch_klipper()


def _emergency_disable(pump_id: str) -> None:
    """Fire-and-forget motor disable for emergency shutdown."""
    try:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.settimeout(2.0)
        _sock.connect(SOCKET_PATH)
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


def _stdin_has_data() -> bool:
    """Check if stdin has data ready (non-blocking)."""
    try:
        return bool(select.select([sys.stdin], [], [], 0.0)[0])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pump control helpers
# ---------------------------------------------------------------------------


def pump_enable(sock: socket.socket, pump_id: str) -> None:
    """Enable the pump stepper driver."""
    _raw_gcode(sock, f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=1")


def pump_disable(sock: socket.socket, pump_id: str) -> None:
    """Disable the pump stepper driver."""
    _raw_gcode(sock, f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=0")


def pump_set_position(sock: socket.socket, pump_id: str, pos: float) -> None:
    """Set the pump's logical position without moving."""
    _raw_gcode(
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
    return _raw_gcode(sock, cmd, timeout=timeout)


def pump_home(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
) -> bool:
    """Home the pump to its limit switch and back off.

    Uses ``homing_direction`` from config to determine which way to
    move toward the switch (+1 = positive, -1 = negative).  After the
    endstop triggers, sets position to 0 and backs off in the opposite
    direction.

    Returns True on success.
    """
    travel = syringe.plunger_travel_mm
    homing_speed = pump_cfg.homing_speed_mm_s
    backoff = pump_cfg.home_backoff_mm
    h_dir = pump_cfg.homing_direction  # +1 or -1

    # Move toward limit switch (overshoot by 5 mm beyond full travel)
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

    # Back off AWAY from the switch (opposite of homing direction)
    if backoff > 0:
        backoff_pos = -h_dir * backoff
        print(f"    Backing off {backoff:.1f} mm ...")
        pump_move(sock, pump_id, position=backoff_pos, speed=1.0, accel=50.0)
        pump_set_position(sock, pump_id, 0.0)
        print("    Position reset to 0.0 mm (after backoff)")

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
    _drain_socket(sock)
    _raw_gcode(sock, "QUERY_ENDSTOPS")
    time.sleep(0.3)
    _drain_socket(sock)

    resp = _raw_send(
        sock,
        "objects/query",
        {"objects": {"query_endstops": ["last_query"]}},
        timeout=3.0,
    )

    if debug:
        print(f"\n  [DEBUG] raw response: {json.dumps(resp, indent=2)}")

    status = resp.get("result", {}).get("status", {})
    last_query = status.get("query_endstops", {}).get("last_query", {})

    # manual_stepper endstop keys look like "manual_stepper pump_0"
    for key, raw_state in last_query.items():
        if pump_id in key:
            if isinstance(raw_state, int):
                return "TRIGGERED" if raw_state else "open"
            return str(raw_state)

    return "???"


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def phase_spin(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    pumps_cfg: PumpsConfig,
    speed: float,
    cycles: int,
) -> None:
    """Phase 1: basic motor spin test.

    Moves forward one revolution (rotation_distance = 4 mm) and back.
    Verifies motor wiring and direction.
    """
    rot_dist = pumps_cfg.stepper.rotation_distance
    pause_s = pumps_cfg.stepper.direction_reversal_pause_s
    pause_ms = int(pause_s * 1000)

    print()
    print("=" * 60)
    print(f"  PHASE 1: MOTOR SPIN TEST ({pump_id})")
    print(f"  {pump_cfg.octopus_slot} -- step={pump_cfg.pins.step} "
          f"dir={pump_cfg.pins.dir} enable={pump_cfg.pins.enable}")
    print(f"  rotation_distance={rot_dist} mm, speed={speed} mm/s")
    print("=" * 60)
    print()

    pump_set_position(sock, pump_id, 0.0)

    for i in range(1, cycles + 1):
        print(f"    Cycle {i}/{cycles}: forward {rot_dist} mm "
              f"(1 rev) ...", end="", flush=True)
        pump_move(sock, pump_id, position=rot_dist, speed=speed)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

        print(f"    Cycle {i}/{cycles}: reverse {rot_dist} mm "
              f"(1 rev) ...", end="", flush=True)
        pump_move(sock, pump_id, position=0.0, speed=speed)
        print(" done")

        if i < cycles:
            _raw_gcode(sock, f"G4 P{pause_ms}")

    print()
    print("  Motor spin test complete.")
    print("  Verify: motor turned smoothly in both directions.")
    print()


def phase_endstop(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
) -> bool:
    """Phase 2: interactive limit switch verification.

    Polls the endstop and auto-advances when an OPEN -> TRIGGERED
    transition is detected.  Press Enter to skip.

    Returns True if the test passed.
    """
    print()
    print("=" * 60)
    print(f"  PHASE 2: LIMIT SWITCH VERIFICATION ({pump_id})")
    print(f"  Endstop pin: {pump_cfg.endstop_polarity}"
          f"{pump_cfg.endstop_pin}")
    print("=" * 60)
    print()
    print("  Release the limit switch, then press it.")
    print("  The test auto-advances on OPEN -> TRIGGERED.")
    print("  Press Enter to skip.")
    print()

    saw_open = False
    saw_triggered = False

    while True:
        state = query_pump_endstop(sock, pump_id)

        if state.lower() == "open":
            saw_open = True
            indicator = "[ OPEN     ]"
        elif state.upper() == "TRIGGERED":
            saw_triggered = True
            indicator = "[TRIGGERED!]"
        else:
            indicator = f"[{state:^11s}]"

        sys.stdout.write(f"\r     {pump_id}: {indicator}    ")
        sys.stdout.flush()

        if saw_open and saw_triggered:
            print(f"\n     PASS  {pump_id} limit switch")
            print()
            return True

        if _stdin_has_data():
            sys.stdin.readline()
            print()
            if saw_open and not saw_triggered:
                print(f"     FAIL  {pump_id}: saw OPEN but never TRIGGERED")
            elif saw_triggered and not saw_open:
                print(f"     FAIL  {pump_id}: always TRIGGERED, never OPEN")
            else:
                print(f"     SKIP  {pump_id}")
            print()
            return False

        time.sleep(0.3)


def phase_home(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
) -> bool:
    """Phase 3: home the pump to its limit switch.

    Returns True on success.
    """
    print()
    print("=" * 60)
    print(f"  PHASE 3: HOMING ({pump_id})")
    print(f"  Speed: {pump_cfg.homing_speed_mm_s} mm/s, "
          f"backoff: {pump_cfg.home_backoff_mm} mm")
    print("=" * 60)
    print()

    ok = pump_home(sock, pump_id, pump_cfg, syringe)
    if ok:
        print("    Homing complete. Position = 0.0 mm (syringe full)")
    else:
        print("    !! Homing FAILED")
    print()
    return ok


def phase_full_travel(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    speed: float,
) -> None:
    """Phase 4: full plunger stroke test.

    Moves from 0 to plunger_travel_mm (dispense all) and back to 0
    (retract all).  Direction is determined by ``homing_direction``:
    dispensing moves away from the limit switch.
    """
    travel = syringe.plunger_travel_mm
    vol = syringe.volume_ml
    dsign = -pump_cfg.homing_direction
    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )

    print()
    print("=" * 60)
    print(f"  PHASE 4: FULL TRAVEL TEST ({pump_id})")
    print(f"  Stroke: {travel} mm = {vol} ml, speed: {speed} mm/s")
    print("=" * 60)
    print()

    dispense_pos = dsign * travel
    print(f"    Dispensing: 0 -> {dispense_pos:.1f} mm ({vol} ml) ...",
          end="", flush=True)
    pump_move(sock, pump_id, position=dispense_pos, speed=speed)
    print(" done")

    _raw_gcode(sock, f"G4 P{pause_ms}")

    print(f"    Retracting: {dispense_pos:.1f} -> 0 mm ({vol} ml) ...",
          end="", flush=True)
    pump_move(sock, pump_id, position=0.0, speed=speed)
    print(" done")

    print()
    print("  Full travel test complete.")
    print(f"  Verify: plunger moved {travel} mm forward then back.")
    print()


def phase_continuous_cycle(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    speed: float,
    cycles: int,
) -> None:
    """Phase 5: continuous dispense/retract cycles.

    Slowly dispenses the full syringe volume, dwells, retracts, dwells,
    and repeats.  During retract the syringe is assumed to be connected
    to a water supply (no valve control needed yet).
    """
    travel = syringe.plunger_travel_mm
    vol = syringe.volume_ml
    dsign = -pump_cfg.homing_direction
    dispense_pos = dsign * travel
    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )

    print()
    print("=" * 60)
    print(f"  PHASE 5: CONTINUOUS CYCLE TEST ({pump_id})")
    print(f"  {vol} ml per cycle, {cycles} cycles, {speed} mm/s")
    print(f"  NOTE: retract assumes water supply connected (no valves)")
    print("=" * 60)
    print()

    for i in range(1, cycles + 1):
        print(f"    Cycle {i}/{cycles}:")

        print(f"      Dispensing {vol} ml ({travel} mm) ...", end="", flush=True)
        pump_move(sock, pump_id, position=dispense_pos, speed=speed)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

        retract_speed = min(speed * 1.5, pump_cfg.max_retract_speed_mm_s)
        print(f"      Retracting {vol} ml ({travel} mm) at "
              f"{retract_speed:.1f} mm/s ...", end="", flush=True)
        pump_move(sock, pump_id, position=0.0, speed=retract_speed)
        print(" done")

        if i < cycles:
            _raw_gcode(sock, "G4 P1000")

    print()
    print(f"  Continuous cycle test complete ({cycles} cycles).")
    print()


def phase_precise_volume(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    volume_ml: float,
    speed: float,
) -> None:
    """Phase 6: dispense a precise volume.

    Homes the pump, then dispenses exactly the requested volume.
    """
    travel_mm = volume_to_mm(volume_ml, syringe)

    if travel_mm > syringe.plunger_travel_mm:
        print(f"  !! Requested volume {volume_ml} ml exceeds syringe "
              f"capacity {syringe.volume_ml} ml.  Clamping.")
        volume_ml = syringe.volume_ml
        travel_mm = syringe.plunger_travel_mm

    dsign = -pump_cfg.homing_direction
    dispense_pos = dsign * travel_mm

    print()
    print("=" * 60)
    print(f"  PHASE 6: PRECISE VOLUME DISPENSE ({pump_id})")
    print(f"  Target: {volume_ml:.3f} ml = {travel_mm:.4f} mm")
    print(f"  Speed: {speed} mm/s")
    print("=" * 60)
    print()

    print(f"    Dispensing {volume_ml:.3f} ml ...", end="", flush=True)
    pump_move(sock, pump_id, position=dispense_pos, speed=speed)
    print(" done")

    print()
    print(f"  Dispensed {volume_ml:.3f} ml "
          f"(plunger at {dispense_pos:.4f} mm).")

    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )
    _raw_gcode(sock, f"G4 P{pause_ms}")

    retract_speed = pump_cfg.max_retract_speed_mm_s
    print(f"    Retracting to 0 at {retract_speed:.1f} mm/s ...",
          end="", flush=True)
    pump_move(sock, pump_id, position=0.0, speed=retract_speed)
    print(" done")
    print()


def phase_microdose(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    speed: float,
) -> None:
    """Phase 7: micro-dose precision test.

    Dispenses tiny volumes (0.01, 0.05, 0.1 ml) to exercise the high
    microstepping resolution.  Returns to home between each dose.
    """
    doses_ml = [0.01, 0.05, 0.1]
    dsign = -pump_cfg.homing_direction
    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )

    print()
    print("=" * 60)
    print(f"  PHASE 7: MICRO-DOSE PRECISION TEST ({pump_id})")
    print(f"  Doses: {doses_ml} ml")
    microstep_res = (
        _CFG.pumps.stepper.rotation_distance  # type: ignore[union-attr]
        / _CFG.pumps.stepper.driver_pulses_per_rev  # type: ignore[union-attr]
    )
    print(f"  Microstep resolution: {microstep_res * 1000:.4f} um")
    print("=" * 60)
    print()

    for dose in doses_ml:
        travel_mm = volume_to_mm(dose, syringe)
        microsteps_needed = travel_mm / microstep_res

        print(f"    Dose: {dose:.3f} ml = {travel_mm:.4f} mm "
              f"({microsteps_needed:.0f} microsteps)")

        dispense_pos = dsign * travel_mm
        print(f"      Dispensing ...", end="", flush=True)
        pump_move(sock, pump_id, position=dispense_pos, speed=speed)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

        print(f"      Retracting ...", end="", flush=True)
        pump_move(sock, pump_id, position=0.0, speed=speed)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

    print()
    print("  Micro-dose test complete.")
    print("  Verify: each dose produced a visible, proportional output.")
    print()


def phase_speed_ramp(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
) -> None:
    """Phase 8: speed ramp test.

    Dispenses 0.5 ml at progressively higher speeds to find the reliable
    operating range.  Listen for stalling or rough motion.
    """
    dose_ml = 0.5
    travel_mm = volume_to_mm(dose_ml, syringe)
    dsign = -pump_cfg.homing_direction
    dispense_pos = dsign * travel_mm
    max_spd = pump_cfg.max_dispense_speed_mm_s
    speeds = [s for s in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0] if s <= max_spd]
    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )
    retract_speed = pump_cfg.max_retract_speed_mm_s

    print()
    print("=" * 60)
    print(f"  PHASE 8: SPEED RAMP TEST ({pump_id})")
    print(f"  Dose per speed: {dose_ml} ml = {travel_mm:.2f} mm")
    print(f"  Speeds: {speeds} mm/s")
    print(f"  (Ctrl-C to abort if motor stalls)")
    print("=" * 60)
    print()

    for spd in speeds:
        duration_s = travel_mm / spd
        flow_rate_ml_s = dose_ml / duration_s

        print(f"    --- {spd} mm/s "
              f"({flow_rate_ml_s:.3f} ml/s, ~{duration_s:.1f} s) ---")

        print(f"      Dispensing ...", end="", flush=True)
        pump_move(sock, pump_id, position=dispense_pos, speed=spd)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

        print(f"      Retracting ...", end="", flush=True)
        pump_move(sock, pump_id, position=0.0, speed=retract_speed)
        print(" done")

        _raw_gcode(sock, f"G4 P{pause_ms}")

    print()
    print("  Speed ramp test complete.")
    print("  Note the highest speed with smooth, quiet motion.")
    print()


def phase_repeatability(
    sock: socket.socket,
    pump_id: str,
    pump_cfg: PumpMotorConfig,
    syringe: SyringeConfig,
    speed: float,
    cycles: int,
) -> None:
    """Phase 9: repeatability / step-loss check.

    Runs N full dispense/retract cycles at working speed, then re-homes.
    If steps were lost, the re-home position will differ from the
    original home, indicating the plunger drifted.
    """
    travel = syringe.plunger_travel_mm
    vol = syringe.volume_ml
    dsign = -pump_cfg.homing_direction
    dispense_pos = dsign * travel
    pause_ms = int(
        _CFG.pumps.stepper.direction_reversal_pause_s * 1000  # type: ignore[union-attr]
    )
    retract_speed = min(speed * 1.5, pump_cfg.max_retract_speed_mm_s)

    print()
    print("=" * 60)
    print(f"  PHASE 9: REPEATABILITY TEST ({pump_id})")
    print(f"  {cycles} full cycles ({vol} ml each), then re-home")
    print(f"  Speed: {speed} mm/s dispense, {retract_speed:.1f} mm/s retract")
    print("=" * 60)
    print()

    for i in range(1, cycles + 1):
        pct = (i / cycles) * 100
        sys.stdout.write(
            f"\r    Cycle {i}/{cycles} ({pct:.0f}%) ..."
        )
        sys.stdout.flush()

        pump_move(sock, pump_id, position=dispense_pos, speed=speed)
        _raw_gcode(sock, f"G4 P{pause_ms}")

        pump_move(sock, pump_id, position=0.0, speed=retract_speed)
        if i < cycles:
            _raw_gcode(sock, f"G4 P{pause_ms}")

    print(" done")
    print()

    print("    Re-homing to check for step loss ...")
    ok = pump_home(sock, pump_id, pump_cfg, syringe)
    if ok:
        print("    Re-home succeeded.  If the plunger is at the same")
        print("    physical position as the original home, no steps were lost.")
    else:
        print("    !! Re-home failed -- manual inspection needed.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _get_pump_config(
    pump_id: str,
) -> tuple[PumpsConfig, PumpMotorConfig, SyringeConfig]:
    """Look up pump configuration by ID.

    Returns
    -------
    tuple
        (PumpsConfig, PumpMotorConfig, SyringeConfig) for the pump.

    Raises
    ------
    SystemExit
        If pumps are disabled or the pump ID is not found.
    """
    if _CFG.pumps is None:
        print("  !! Pumps are not enabled in machine.yaml")
        sys.exit(1)

    pumps_cfg = _CFG.pumps
    if pump_id not in pumps_cfg.motors:
        print(f"  !! Pump '{pump_id}' not found in config. "
              f"Available: {list(pumps_cfg.motors.keys())}")
        sys.exit(1)

    pump_motor = pumps_cfg.motors[pump_id]
    syringe = pump_motor.syringe
    return pumps_cfg, pump_motor, syringe


# Map of test name -> phase function signature style
_TEST_NAMES = [
    "spin", "endstop", "home", "travel", "cycle",
    "dispense", "microdose", "speedramp", "repeatability",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Syringe pump hardware bring-up and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pump", default="pump_0",
        help="Pump identifier (default: pump_0)",
    )
    parser.add_argument(
        "--test", choices=_TEST_NAMES, default=None,
        help="Run a single test phase (default: run all)",
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Override dispense speed in mm/s (default: from config)",
    )
    parser.add_argument(
        "--volume", type=float, default=0.5,
        help="Target volume for precise dispense test in ml (default: 0.5)",
    )
    parser.add_argument(
        "--cycles", type=int, default=5,
        help="Repeat count for cycle/repeatability tests (default: 5)",
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't overwrite printer.cfg (assume correct config is loaded)",
    )
    parser.add_argument(
        "--skip-home", action="store_true",
        help="Skip homing (assume pump is already homed)",
    )
    args = parser.parse_args()

    pump_id = args.pump
    pumps_cfg, pump_motor, syringe = _get_pump_config(pump_id)

    speed = args.speed if args.speed is not None else pump_motor.max_dispense_speed_mm_s
    ps = pumps_cfg.stepper

    # Derived values for display
    microstep_res_um = (ps.rotation_distance / ps.driver_pulses_per_rev) * 1000
    vol_per_mm = syringe.volume_per_mm
    mm_per_ml = syringe.mm_per_ml

    print("=" * 60)
    print("  SYRINGE PUMP TEST  (pump-only, no XYZ motion)")
    print("=" * 60)
    print()
    print("  Config: kinematics=none -- only pump stepper is defined.")
    print("  No X, Y, or Z axes are configured or homed.")
    print()
    print(f"  Pump:             {pump_id} ({pump_motor.octopus_slot})")

    print(f"  Pins:             step={pump_motor.pins.step} "
          f"dir={pump_motor.pins.dir} enable={pump_motor.pins.enable}")
    print(f"  Endstop:          {pump_motor.endstop_polarity}"
          f"{pump_motor.endstop_pin}")
    h_dir_label = "positive (+)" if pump_motor.homing_direction > 0 else "negative (-)"
    print(f"  Homing direction: {h_dir_label} (switch side)")
    print()
    print(f"  Motor parameters:")
    print(f"    type:             {ps.motor_type} "
          f"({ps.full_steps_per_rotation} full steps/rev)")
    print(f"    microsteps:       {ps.klipper_microsteps} "
          f"({ps.driver_pulses_per_rev} pulses/rev)")
    print(f"    rotation_dist:    {ps.rotation_distance} mm/rev "
          f"(lead screw)")
    print(f"    step_pulse:       "
          f"{ps.step_pulse_duration_s * 1e6:.0f} us")
    print(f"    resolution:       {microstep_res_um:.4f} um/microstep")
    print()
    print(f"  Syringe:")
    print(f"    capacity:         {syringe.volume_ml} ml")
    print(f"    plunger stroke:   {syringe.plunger_travel_mm} mm")
    print(f"    volume/mm:        {vol_per_mm:.4f} ml/mm")
    print(f"    mm/ml:            {mm_per_ml:.1f} mm/ml")
    print()
    print(f"  Test speed:         {speed} mm/s")
    print(f"  Direction pause:    "
          f"{ps.direction_reversal_pause_s * 1000:.0f} ms")
    print()

    # --- Write printer.cfg and restart Klipper ---
    if not args.no_config_write:
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            print(f"  Backed up existing printer.cfg -> {backup}")

        config_text = generate_pump_test_cfg(_CFG, pump_ids=[pump_id])
        PRINTER_CFG_PATH.write_text(config_text)
        print(f"  Wrote pump test printer.cfg to {PRINTER_CFG_PATH}")
        print()

        print("  Restarting Klipper to load pump test config...")
        restart_klipper()

    print("  Waiting for Klipper to become ready...")
    sock = _wait_for_ready(timeout=45.0)
    print("  [OK] Klipper is ready")
    print()

    # Determine which tests to run
    single_test = args.test
    run_all = single_test is None

    try:
        # For tests that need homing, decide whether to home
        need_home = run_all or single_test in (
            "home", "travel", "cycle", "dispense",
            "microdose", "speedramp", "repeatability",
        )
        homed = False

        # --- Phase 1: Motor spin ---
        if run_all or single_test == "spin":
            pump_enable(sock, pump_id)
            phase_spin(sock, pump_id, pump_motor, pumps_cfg, speed, args.cycles)

        # --- Phase 2: Limit switch ---
        if run_all or single_test == "endstop":
            endstop_ok = phase_endstop(sock, pump_id, pump_motor)
            if run_all and not endstop_ok:
                print("  Endstop verification failed -- skipping homing.")
                need_home = False

        # --- Phase 3: Homing ---
        if need_home and not args.skip_home:
            pump_enable(sock, pump_id)
            homed = phase_home(sock, pump_id, pump_motor, syringe)
            if run_all and not homed:
                print("  Homing failed -- skipping remaining travel tests.")
        elif args.skip_home:
            homed = True
            print("  [SKIP] Homing (--skip-home)")
            print()

        if single_test == "home":
            pass  # Already ran homing above
        elif homed or single_test in ("spin", "endstop"):
            # --- Phase 4: Full travel ---
            if run_all or single_test == "travel":
                phase_full_travel(
                    sock, pump_id, pump_motor, syringe, speed,
                )

            # --- Phase 5: Continuous cycle ---
            if run_all or single_test == "cycle":
                phase_continuous_cycle(
                    sock, pump_id, pump_motor, syringe, speed, args.cycles,
                )

            # --- Phase 6: Precise volume ---
            if run_all or single_test == "dispense":
                phase_precise_volume(
                    sock, pump_id, pump_motor, syringe,
                    args.volume, speed,
                )

            # --- Phase 7: Micro-dose ---
            if run_all or single_test == "microdose":
                phase_microdose(
                    sock, pump_id, pump_motor, syringe, speed,
                )

            # --- Phase 8: Speed ramp ---
            if run_all or single_test == "speedramp":
                phase_speed_ramp(
                    sock, pump_id, pump_motor, syringe,
                )

            # --- Phase 9: Repeatability ---
            if run_all or single_test == "repeatability":
                phase_repeatability(
                    sock, pump_id, pump_motor, syringe,
                    speed, args.cycles,
                )

        # Disable motor after all tests
        pump_disable(sock, pump_id)

    except KeyboardInterrupt:
        print("\n\n  Interrupted -- disabling pump motor...")
        try:
            payload = (
                json.dumps({
                    "id": 9999,
                    "method": "gcode/script",
                    "params": {
                        "script": (
                            f"MANUAL_STEPPER STEPPER={pump_id} ENABLE=0"
                        ),
                    },
                }).encode()
                + ETX
            )
            sock.settimeout(1.0)
            sock.sendall(payload)
        except Exception:
            pass

    finally:
        sock.close()

    # --- Summary ---
    print()
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print()
    print("  Checklist:")
    print(f"    [ ] Motor spin: {pump_id} turned smoothly both directions?")
    print(f"    [ ] Limit switch: open when released, TRIGGERED when pressed?")
    print(f"    [ ] Homing: plunger stopped at limit switch?")
    print(f"    [ ] Full travel: plunger moved {syringe.plunger_travel_mm} mm "
          f"({syringe.volume_ml} ml) and back?")
    print(f"    [ ] Continuous cycle: smooth dispense/retract over "
          f"{args.cycles} cycles?")
    print(f"    [ ] Precise volume: correct amount dispensed?")
    print(f"    [ ] Micro-dose: proportional tiny outputs?")
    print(f"    [ ] Speed ramp: highest smooth speed noted?")
    print(f"    [ ] Repeatability: no step loss after multiple cycles?")
    print()

    if not args.no_config_write:
        backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
        if backup.exists():
            print(f"  To restore your production config:")
            print(f"    mv {backup} {PRINTER_CFG_PATH}")
            print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        _emergency_disable("pump_0")
        sys.exit(1)
