#!/usr/bin/env python3
"""Interactive pump testbed for CMY + purge syringe pump system.

Menu-driven script for testing, controlling, and calibrating the four
syringe pumps (Cyan, Magenta, Yellow, Purge/IPA).  Axis motors are
disabled by default -- only pump steppers are active.

All machine parameters come from ``machine.yaml``; nothing is hardcoded.

Usage::

    .venv/bin/python robot_control/scripts/pump_testbed.py
    .venv/bin/python robot_control/scripts/pump_testbed.py --no-config-write
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import (
    MachineConfig,
    PumpMotorConfig,
    PumpsConfig,
    SyringeConfig,
    load_config,
)
from robot_control.configs.printer_cfg import generate_pump_test_cfg
from robot_control.hardware.pump_control import (
    PRINTER_CFG_PATH,
    emergency_disable,
    mm_to_volume,
    pump_disable,
    pump_enable,
    pump_home,
    pump_home_with_backlash,
    pump_move,
    pump_set_position,
    query_pump_endstop,
    raw_gcode,
    restart_klipper,
    set_pin,
    stdin_has_data,
    volume_to_mm,
    wait_for_ready,
)
from src.utils.fs import atomic_yaml_dump, load_yaml

_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "machine.yaml"
_MANUAL_CAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs" / "sim" / "manual_calibration_results.yaml"
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class SessionState:
    """Tracks per-pump homing state, valve state, and the active socket."""

    def __init__(self, cfg: MachineConfig) -> None:
        self.cfg = cfg
        self.pumps_cfg: PumpsConfig = cfg.pumps  # type: ignore[assignment]
        self.socket_path: str = cfg.connection.socket_path
        self.sock: socket.socket | None = None
        self.homed: dict[str, bool] = {
            pid: False for pid in self.pumps_cfg.motors
        }
        self.valve_open_flag: bool = False

    @property
    def pump_ids(self) -> list[str]:
        return list(self.pumps_cfg.motors.keys())

    @property
    def ink_pump_ids(self) -> list[str]:
        """Pump IDs that carry ink (not purge)."""
        return [
            pid for pid, m in self.pumps_cfg.motors.items()
            if m.fluid != "purge"
        ]

    @property
    def purge_pump_id(self) -> str | None:
        for pid, m in self.pumps_cfg.motors.items():
            if m.fluid == "purge":
                return pid
        return None

    def motor(self, pid: str) -> PumpMotorConfig:
        return self.pumps_cfg.motors[pid]

    def syringe(self, pid: str) -> SyringeConfig:
        return self.motor(pid).syringe

    def pump_label(self, pid: str) -> str:
        m = self.motor(pid)
        return f"{pid} ({m.fluid}, {m.octopus_slot})"

    def ensure_connected(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("Not connected to Klipper")
        return self.sock

    @property
    def _valve_output(self) -> str:
        return self.pumps_cfg.refill_valve_output

    @property
    def _valve_delay(self) -> float:
        return self.pumps_cfg.refill_valve_delay_s

    def open_valve(self) -> None:
        """Open the refill valve and wait for the servo to transit."""
        name = self._valve_output
        if not name:
            return
        sock = self.ensure_connected()
        print(f"  Opening refill valve ({name}) ...", end="", flush=True)
        set_pin(sock, name, 1)
        time.sleep(self._valve_delay)
        self.valve_open_flag = True
        print(" done")

    def close_valve(self) -> None:
        """Close the refill valve and wait for the servo to transit."""
        name = self._valve_output
        if not name:
            return
        sock = self.ensure_connected()
        print(f"  Closing refill valve ({name}) ...", end="", flush=True)
        set_pin(sock, name, 0)
        time.sleep(self._valve_delay)
        self.valve_open_flag = False
        print(" done")

    def check_travel_limit(
        self,
        pid: str,
        target_position: float,
    ) -> bool:
        """Return True if *target_position* is within plunger travel.

        When ``enforce_travel_limits`` is False or the pump is not homed,
        always returns True (no enforcement).
        """
        if not self.pumps_cfg.enforce_travel_limits:
            return True
        if not self.homed[pid]:
            return True
        sy = self.syringe(pid)
        if abs(target_position) > sy.plunger_travel_mm:
            print(
                f"  !! LIMIT VIOLATION: {pid} target {target_position:.4f} mm "
                f"exceeds plunger travel {sy.plunger_travel_mm:.4f} mm. "
                f"Move rejected."
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _input(prompt: str, default: str = "") -> str:
    """Prompt with optional default shown in brackets."""
    if default:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {prompt}: ").strip()


def _input_float(prompt: str, default: float) -> float:
    raw = _input(prompt, f"{default:g}")
    try:
        return float(raw)
    except ValueError:
        print(f"    Invalid number, using default {default}")
        return default


def _input_int(prompt: str, default: int) -> int:
    raw = _input(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"    Invalid number, using default {default}")
        return default


def _input_bool(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _input(f"{prompt} ({hint})", "").lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _select_pumps(
    state: SessionState,
    prompt: str = "Which pumps?",
    allow_all: bool = True,
) -> list[str]:
    """Prompt user to select one or more pumps by number."""
    ids = state.pump_ids
    print()
    for i, pid in enumerate(ids):
        print(f"    {i + 1}. {state.pump_label(pid)}")
    if allow_all:
        print(f"    A. All pumps")
    raw = _input(prompt, "A" if allow_all else "1")
    if raw.upper() == "A" and allow_all:
        return list(ids)
    selected: list[str] = []
    for token in raw.replace(",", " ").split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(ids):
                selected.append(ids[idx])
        except ValueError:
            pass
    if not selected:
        print("    No valid selection, defaulting to all")
        return list(ids)
    return selected


def _select_single_pump(
    state: SessionState,
    prompt: str = "Which pump?",
) -> str:
    """Prompt for a single pump selection."""
    ids = state.pump_ids
    print()
    for i, pid in enumerate(ids):
        print(f"    {i + 1}. {state.pump_label(pid)}")
    raw = _input(prompt, "1")
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(ids):
            return ids[idx]
    except ValueError:
        pass
    print(f"    Invalid selection, defaulting to {ids[0]}")
    return ids[0]


def _check_homing(
    state: SessionState,
    pump_ids: list[str],
    force_prompt: bool = False,
) -> bool:
    """Check if pumps are homed; offer to home if not.

    Returns
    -------
    bool
        True if safe to proceed (all homed, or user chose to skip).
        False **only** when homing was attempted and failed -- caller
        must abort the current operation.
    """
    not_homed = [pid for pid in pump_ids if not state.homed[pid]]
    if not not_homed and not force_prompt:
        return True
    if not_homed:
        labels = ", ".join(state.pump_label(pid) for pid in not_homed)
        print(f"\n  Not homed: {labels}")
        if _input_bool("Home them now?"):
            ok = _do_homing(state, not_homed)
            if not ok:
                print("  Homing failed. Aborting operation.")
                return False
            return True
        print("  Proceeding without homing -- ensure enough travel space.")
    return True


def _pause_ms(state: SessionState) -> int:
    return int(state.pumps_cfg.stepper.direction_reversal_pause_s * 1000)


def _dwell(state: SessionState) -> None:
    raw_gcode(state.ensure_connected(), f"G4 P{_pause_ms(state)}")


def _dispense_sign(pump_cfg: PumpMotorConfig) -> int:
    """Dispense moves away from the limit switch."""
    return -pump_cfg.homing_direction


# ---------------------------------------------------------------------------
# Control actions
# ---------------------------------------------------------------------------


def _do_homing(state: SessionState, pump_ids: list[str]) -> bool:
    """Home selected pumps with backlash purge.

    The refill valve is opened before homing starts so that fluid can
    enter the syringe during plunger retraction (backoff).  It is
    closed again after all pumps are done.  Motors are disabled after
    homing (both on success and failure).
    """
    sock = state.ensure_connected()
    backlash = state.pumps_cfg.backlash_purge_mm
    all_ok = True

    state.open_valve()

    try:
        for pid in pump_ids:
            m = state.motor(pid)
            sy = state.syringe(pid)
            print(f"\n  Homing {state.pump_label(pid)} ...")
            pump_enable(sock, pid)
            ok = pump_home_with_backlash(sock, pid, m, sy, backlash)
            if ok:
                state.homed[pid] = True
                print(f"  {pid} homed successfully.")
            else:
                print(f"  !! {pid} homing FAILED")
                all_ok = False
            pump_disable(sock, pid)
    finally:
        state.close_valve()

    return all_ok


def _do_dispense(
    state: SessionState,
    pid: str,
    volume_ml: float,
    speed: float,
) -> None:
    """Dispense a volume, retract, then disable the motor."""
    sock = state.ensure_connected()
    m = state.motor(pid)
    sy = state.syringe(pid)
    travel_mm = volume_to_mm(volume_ml, sy)

    if travel_mm > sy.plunger_travel_mm:
        print(f"  !! Volume {volume_ml:.3f} ml exceeds capacity "
              f"{sy.volume_ml:.3f} ml. Clamping.")
        volume_ml = sy.volume_ml
        travel_mm = sy.plunger_travel_mm

    dsign = _dispense_sign(m)
    pos = dsign * travel_mm

    if not state.check_travel_limit(pid, pos):
        return

    pump_enable(sock, pid)
    pump_set_position(sock, pid, 0.0)
    print(f"    Dispensing {volume_ml:.3f} ml ({travel_mm:.4f} mm) "
          f"at {speed:.2f} mm/s ...", end="", flush=True)
    pump_move(sock, pid, position=pos, speed=speed)
    print(" done")

    _dwell(state)

    retract_speed = m.max_retract_speed_mm_s
    print(f"    Retracting at {retract_speed:.1f} mm/s ...",
          end="", flush=True)
    pump_move(sock, pid, position=0.0, speed=retract_speed)
    print(" done")

    pump_disable(sock, pid)


def _do_dispense_no_retract(
    state: SessionState,
    pid: str,
    volume_ml: float,
    speed: float,
    disable_after: bool = True,
) -> float:
    """Dispense a volume without retracting. Returns actual travel mm.

    Parameters
    ----------
    disable_after : bool
        If True (default), disable the stepper after dispensing.
        Set False when the caller will issue further moves on the
        same pump before disabling.

    Returns 0.0 if the move was rejected by the travel limit check.
    """
    sock = state.ensure_connected()
    m = state.motor(pid)
    sy = state.syringe(pid)
    travel_mm = volume_to_mm(volume_ml, sy)

    if travel_mm > sy.plunger_travel_mm:
        volume_ml = sy.volume_ml
        travel_mm = sy.plunger_travel_mm

    dsign = _dispense_sign(m)
    pos = dsign * travel_mm

    if not state.check_travel_limit(pid, pos):
        return 0.0

    pump_enable(sock, pid)
    pump_set_position(sock, pid, 0.0)
    print(f"    Dispensing {volume_ml:.3f} ml ({travel_mm:.4f} mm) "
          f"at {speed:.2f} mm/s ...", end="", flush=True)
    pump_move(sock, pid, position=pos, speed=speed)
    print(" done")

    if disable_after:
        pump_disable(sock, pid)
    return travel_mm


def _do_simultaneous_dispense(
    state: SessionState,
    pump_ids: list[str],
    volumes: dict[str, float],
    speed: float,
    retract: bool = True,
) -> None:
    """Dispense multiple pumps so they all start and finish together.

    Each pump dispenses a different volume.  To finish simultaneously
    the per-pump speed is scaled so every pump takes the same duration
    (governed by the pump with the longest travel at the given speed).

    Parameters
    ----------
    pump_ids : list[str]
        Pumps to run.
    volumes : dict[str, float]
        Target volume per pump in ml.
    speed : float
        Reference speed in mm/s.  The slowest pump (longest travel)
        runs at this speed; faster pumps are slowed proportionally.
    retract : bool
        If True, retract all pumps simultaneously after dispensing.
    """
    sock = state.ensure_connected()

    travels: dict[str, float] = {}
    for pid in pump_ids:
        sy = state.syringe(pid)
        t = volume_to_mm(volumes[pid], sy)
        if t > sy.plunger_travel_mm:
            t = sy.plunger_travel_mm
        m = state.motor(pid)
        dsign = _dispense_sign(m)
        if not state.check_travel_limit(pid, dsign * t):
            return
        travels[pid] = t

    max_travel = max(travels.values())
    if max_travel <= 0:
        return
    duration_s = max_travel / speed

    print(f"\n  Running pumps simultaneously ({duration_s:.1f} s) ...")
    for pid in pump_ids:
        pump_enable(sock, pid)
        pump_set_position(sock, pid, 0.0)

    for i, pid in enumerate(pump_ids):
        m = state.motor(pid)
        dsign = _dispense_sign(m)
        pos = dsign * travels[pid]
        pump_speed = travels[pid] / duration_s if duration_s > 0 else speed
        is_last = (i == len(pump_ids) - 1)

        print(f"    {state.pump_label(pid)}: "
              f"{volumes[pid]:.3f} ml at {pump_speed:.3f} mm/s",
              end="", flush=True)
        pump_move(sock, pid, position=pos, speed=pump_speed, sync=is_last)
        if is_last:
            print("  done (all)")
        else:
            print("")

    if not retract:
        return

    _dwell(state)

    max_retract = max(travels.values())
    retract_duration = max_retract / speed if speed > 0 else 1.0
    for i, pid in enumerate(pump_ids):
        retract_speed = travels[pid] / retract_duration if retract_duration > 0 else speed
        is_last = (i == len(pump_ids) - 1)
        pump_move(sock, pid, position=0.0, speed=retract_speed, sync=is_last)

    print("  Retract complete.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pump_juggle(state: SessionState) -> None:
    """Test 1: rapid back-and-forth to check motor response."""
    print("\n" + "=" * 60)
    print("  TEST: PUMP JUGGLE")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps to juggle?")
    distance = _input_float("Distance (mm)", 1.0)
    speed = _input_float("Speed (mm/s)", 0.5)
    reps = _input_int("Repetitions", 10)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()

    for pid in selected:
        m = state.motor(pid)
        dsign = _dispense_sign(m)
        pos = dsign * distance
        if not state.check_travel_limit(pid, pos):
            continue
        pump_enable(sock, pid)
        pump_set_position(sock, pid, 0.0)

        print(f"\n  Juggling {state.pump_label(pid)}: "
              f"{distance} mm x {reps} reps at {speed} mm/s")

        for i in range(1, reps + 1):
            pump_move(sock, pid, position=pos, speed=speed)
            pump_move(sock, pid, position=0.0, speed=speed)
            sys.stdout.write(f"\r    Rep {i}/{reps}")
            sys.stdout.flush()

        print("  done")
        pump_disable(sock, pid)

    print("\n  Juggle test complete.")


def test_volume(state: SessionState) -> None:
    """Test 2: dispense a user-specified volume."""
    print("\n" + "=" * 60)
    print("  TEST: VOLUME TEST")
    print("=" * 60)

    pid = _select_single_pump(state, "Which pump?")
    volume = _input_float("Volume (ml)", 0.5)
    speed = _input_float("Speed (mm/s)", 0.5)
    if not _check_homing(state, [pid]):
        return

    _do_dispense(state, pid, volume, speed)
    print("\n  Volume test complete.")


def test_limit_switch(state: SessionState) -> None:
    """Test 3: interactive endstop verification."""
    print("\n" + "=" * 60)
    print("  TEST: LIMIT SWITCH VERIFICATION")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps to test?")
    sock = state.ensure_connected()

    for pid in selected:
        m = state.motor(pid)
        print(f"\n  Testing {state.pump_label(pid)}")
        print(f"  Endstop pin: {m.endstop_polarity}{m.endstop_pin}")
        print("  Release the limit switch, then press it.")
        print("  Auto-advances on OPEN -> TRIGGERED.  Press Enter to skip.\n")

        saw_open = False
        saw_triggered = False

        while True:
            es = query_pump_endstop(sock, pid)
            if es.lower() == "open":
                saw_open = True
                indicator = "[ OPEN     ]"
            elif es.upper() == "TRIGGERED":
                saw_triggered = True
                indicator = "[TRIGGERED!]"
            else:
                indicator = f"[{es:^11s}]"

            sys.stdout.write(f"\r     {pid}: {indicator}    ")
            sys.stdout.flush()

            if saw_open and saw_triggered:
                print(f"\n     PASS  {pid} limit switch\n")
                break

            if stdin_has_data():
                sys.stdin.readline()
                print()
                if saw_open and not saw_triggered:
                    print(f"     FAIL  {pid}: saw OPEN but never TRIGGERED")
                elif saw_triggered and not saw_open:
                    print(f"     FAIL  {pid}: always TRIGGERED, never OPEN")
                else:
                    print(f"     SKIP  {pid}")
                print()
                break

            time.sleep(0.3)

    print("  Limit switch test complete.")


def test_speed_ramp(state: SessionState) -> None:
    """Test 4: dispense at increasing speeds."""
    print("\n" + "=" * 60)
    print("  TEST: SPEED RAMP")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps?")
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()
    dose_ml = 0.5

    for pid in selected:
        m = state.motor(pid)
        sy = state.syringe(pid)
        travel_mm = volume_to_mm(dose_ml, sy)
        dsign = _dispense_sign(m)
        dispense_pos = dsign * travel_mm
        max_spd = m.max_dispense_speed_mm_s
        speeds = [s for s in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0] if s <= max_spd]
        retract_speed = m.max_retract_speed_mm_s

        if not state.check_travel_limit(pid, dispense_pos):
            continue

        print(f"\n  Speed ramp for {state.pump_label(pid)}")
        print(f"  Dose: {dose_ml} ml = {travel_mm:.2f} mm")
        print(f"  Speeds: {speeds} mm/s")
        pump_enable(sock, pid)

        for spd in speeds:
            duration_s = travel_mm / spd
            print(f"\n    --- {spd} mm/s (~{duration_s:.1f} s) ---")

            pump_set_position(sock, pid, 0.0)
            print(f"      Dispensing ...", end="", flush=True)
            pump_move(sock, pid, position=dispense_pos, speed=spd)
            print(" done")
            _dwell(state)

            print(f"      Retracting ...", end="", flush=True)
            pump_move(sock, pid, position=0.0, speed=retract_speed)
            print(" done")
            _dwell(state)

        pump_disable(sock, pid)

    print("\n  Speed ramp test complete.")
    print("  Note the highest speed with smooth, quiet motion.")


def test_repeatability(state: SessionState) -> None:
    """Test 5: N dispense/retract cycles then re-home for step-loss check."""
    print("\n" + "=" * 60)
    print("  TEST: REPEATABILITY (step-loss check)")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps?")
    cycles = _input_int("Cycles", 10)
    speed = _input_float("Dispense speed (mm/s)", 0.5)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()

    for pid in selected:
        m = state.motor(pid)
        sy = state.syringe(pid)
        travel = sy.plunger_travel_mm
        dsign = _dispense_sign(m)
        dispense_pos = dsign * travel
        retract_speed = min(speed * 1.5, m.max_retract_speed_mm_s)

        if not state.check_travel_limit(pid, dispense_pos):
            continue

        print(f"\n  Repeatability for {state.pump_label(pid)}")
        print(f"  {cycles} cycles, {speed} mm/s dispense, "
              f"{retract_speed:.1f} mm/s retract")
        pump_enable(sock, pid)

        for i in range(1, cycles + 1):
            sys.stdout.write(f"\r    Cycle {i}/{cycles} "
                             f"({i / cycles * 100:.0f}%) ...")
            sys.stdout.flush()
            pump_move(sock, pid, position=dispense_pos, speed=speed)
            _dwell(state)
            pump_move(sock, pid, position=0.0, speed=retract_speed)
            if i < cycles:
                _dwell(state)

        print(" done")
        print("    Re-homing to check for step loss ...")
        ok = pump_home(sock, pid, m, sy)
        if ok:
            state.homed[pid] = True
            print("    Re-home OK. Same physical position = no step loss.")
        else:
            print("    !! Re-home failed -- manual inspection needed.")
        pump_disable(sock, pid)

    print("\n  Repeatability test complete.")


def test_full_travel(state: SessionState) -> None:
    """Test 6: full plunger stroke."""
    print("\n" + "=" * 60)
    print("  TEST: FULL TRAVEL")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps?")
    speed = _input_float("Speed (mm/s)", 0.5)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()

    for pid in selected:
        m = state.motor(pid)
        sy = state.syringe(pid)
        travel = sy.plunger_travel_mm
        dsign = _dispense_sign(m)
        dispense_pos = dsign * travel

        if not state.check_travel_limit(pid, dispense_pos):
            continue

        print(f"\n  Full travel for {state.pump_label(pid)}")
        print(f"  Stroke: {travel} mm = {sy.volume_ml} ml")
        pump_enable(sock, pid)

        print(f"    Dispensing: 0 -> {dispense_pos:.1f} mm ...",
              end="", flush=True)
        pump_move(sock, pid, position=dispense_pos, speed=speed)
        print(" done")
        _dwell(state)

        print(f"    Retracting: {dispense_pos:.1f} -> 0 mm ...",
              end="", flush=True)
        pump_move(sock, pid, position=0.0, speed=speed)
        print(" done")

        pump_disable(sock, pid)

    print("\n  Full travel test complete.")


def test_volume_repeatability(state: SessionState) -> None:
    """Test 7: repeated 0.5 ml dispense/retract with visual confirmation."""
    print("\n" + "=" * 60)
    print("  TEST: VOLUME REPEATABILITY")
    print("=" * 60)

    pid = _select_single_pump(state, "Which pump?")
    volume = _input_float("Volume per cycle (ml)", 0.5)
    speed = _input_float("Speed (mm/s)", 0.5)
    if not _check_homing(state, [pid]):
        return

    cycle = 0
    while True:
        cycle += 1
        print(f"\n  --- Cycle {cycle} ---")
        _do_dispense(state, pid, volume, speed)
        print("  Confirm the dispensed volume visually.")
        if not _input_bool("Continue with another cycle?"):
            break
        if _input_bool("Change volume or speed?", default=False):
            volume = _input_float("Volume (ml)", volume)
            speed = _input_float("Speed (mm/s)", speed)

    print(f"\n  Volume repeatability done after {cycle} cycles.")


# ---------------------------------------------------------------------------
# Control actions
# ---------------------------------------------------------------------------


def control_homing(state: SessionState) -> None:
    """Control 1: home selected pumps."""
    print("\n" + "=" * 60)
    print("  CONTROL: HOMING")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps to home?")
    _do_homing(state, selected)


def control_setup(state: SessionState) -> None:
    """Control 2: manual backlash purge."""
    print("\n" + "=" * 60)
    print("  CONTROL: BACKLASH PURGE (SETUP)")
    print("=" * 60)

    backlash = state.pumps_cfg.backlash_purge_mm
    print(f"  Backlash purge distance: {backlash} mm (from config)")

    selected = _select_pumps(state, "Which pumps?")
    sock = state.ensure_connected()

    for pid in selected:
        m = state.motor(pid)
        dsign = _dispense_sign(m)
        purge_pos = dsign * backlash

        print(f"\n  Purging backlash on {state.pump_label(pid)} ...")
        pump_enable(sock, pid)
        pump_set_position(sock, pid, 0.0)
        pump_move(sock, pid, position=purge_pos, speed=1.0, accel=50.0)
        pump_set_position(sock, pid, 0.0)
        pump_disable(sock, pid)
        print(f"  Done. Position reset to 0.0 mm.")

    print("\n  Backlash purge complete.")


def control_individual_pump(state: SessionState) -> None:
    """Control 3: run a single pump (dispense only, no auto-retract)."""
    print("\n" + "=" * 60)
    print("  CONTROL: INDIVIDUAL PUMP")
    print("=" * 60)

    pid = _select_single_pump(state, "Which pump?")
    volume = _input_float("Volume (ml)", 0.5)
    speed = _input_float("Speed (mm/s)", 0.5)
    if not _check_homing(state, [pid]):
        return

    _do_dispense_no_retract(state, pid, volume, speed)
    print("\n  Individual pump run complete.")


def control_multi_pump(state: SessionState) -> None:
    """Control 4: run multiple pumps sequentially or simultaneously."""
    print("\n" + "=" * 60)
    print("  CONTROL: MULTI-PUMP")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps?")
    if not _check_homing(state, selected):
        return

    volumes: dict[str, float] = {}
    for pid in selected:
        volumes[pid] = _input_float(
            f"Volume for {state.pump_label(pid)} (ml)", 0.5
        )
    speed = _input_float("Speed (mm/s)", 0.5)
    simultaneous = _input_bool("Run simultaneously?", default=False)

    sock = state.ensure_connected()

    if simultaneous:
        _do_simultaneous_dispense(state, selected, volumes, speed)
        for pid in selected:
            pump_disable(sock, pid)
    else:
        print("\n  Running pumps sequentially ...")
        for pid in selected:
            print(f"\n  {state.pump_label(pid)}:")
            _do_dispense(state, pid, volumes[pid], speed)

    print("\n  Multi-pump run complete.")


def control_purge(state: SessionState) -> None:
    """Control 5: flush the mixing manifold with IPA (full syringe)."""
    print("\n" + "=" * 60)
    print("  CONTROL: PURGE (manifold flush)")
    print("=" * 60)

    purge_pid = state.purge_pump_id
    if purge_pid is None:
        print("  !! No pump with fluid='purge' found in config.")
        return

    sy = state.syringe(purge_pid)
    volume = sy.volume_ml
    speed = _input_float("Speed (mm/s)", 0.5)
    if not _check_homing(state, [purge_pid]):
        return

    print(f"\n  Purging full syringe ({volume:.2f} ml) "
          f"with {state.pump_label(purge_pid)} ...")
    _do_dispense(state, purge_pid, volume, speed)
    print("\n  Purge complete.")


def control_fill(state: SessionState) -> None:
    """Control 6: retract plunger(s) fully to fill syringes.

    The refill valve is opened automatically so fluid can enter the
    syringe during plunger retraction.
    """
    print("\n" + "=" * 60)
    print("  CONTROL: FILL (retract plunger)")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps to fill?")
    speed = _input_float("Retract speed (mm/s)", 2.0)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()
    state.open_valve()
    try:
        for pid in selected:
            sy = state.syringe(pid)
            print(f"\n  Filling {state.pump_label(pid)} "
                  f"({sy.volume_ml:.2f} ml) ...", end="", flush=True)
            pump_enable(sock, pid)
            pump_move(sock, pid, position=0.0, speed=speed)
            print(" done")
            pump_disable(sock, pid)
    finally:
        state.close_valve()

    print("\n  Fill complete.")


def control_valve_cycle(state: SessionState) -> None:
    """Control 7: full fill/empty cycle with automatic valve control.

    Sequence:
      1. Open refill valve (wait for servo transit)
      2. Retract plungers fully (fill syringes from reservoir)
      3. Close refill valve (wait for servo transit)
      4. Extend plungers fully (empty syringes into manifold)
    """
    print("\n" + "=" * 60)
    print("  CONTROL: VALVE CYCLE (fill + empty)")
    print("=" * 60)

    if not state._valve_output:
        print("  !! No refill_valve_output configured in pumps section.")
        return

    selected = _select_pumps(state, "Which pumps to cycle?")
    speed = _input_float("Speed (mm/s)", 2.0)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()

    # Phase 1: open valve and fill
    state.open_valve()
    try:
        for pid in selected:
            sy = state.syringe(pid)
            print(f"\n  Filling {state.pump_label(pid)} "
                  f"({sy.volume_ml:.2f} ml) ...", end="", flush=True)
            pump_enable(sock, pid)
            pump_move(sock, pid, position=0.0, speed=speed)
            print(" done")
            pump_disable(sock, pid)
    finally:
        state.close_valve()

    _dwell(state)

    # Phase 2: empty (valve closed, fluid exits through nozzle)
    for pid in selected:
        m = state.motor(pid)
        sy = state.syringe(pid)
        dsign = _dispense_sign(m)
        dispense_pos = dsign * sy.plunger_travel_mm
        if not state.check_travel_limit(pid, dispense_pos):
            continue
        print(f"\n  Emptying {state.pump_label(pid)} "
              f"({sy.volume_ml:.2f} ml) ...", end="", flush=True)
        pump_enable(sock, pid)
        pump_move(sock, pid, position=dispense_pos, speed=speed)
        print(" done")
        pump_disable(sock, pid)

    print("\n  Valve cycle complete.")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_volume(state: SessionState) -> None:
    """Calibration 1: volume accuracy with automated valve cycling and CSV log.

    Two modes:

    * **Fixed volume** -- dispense the same volume N times to measure
      repeatability and systematic bias.
    * **Volume sweep** -- dispense 0.1 ml to 1.0 ml (10 steps) to
      characterise linearity across the operating range.

    Each cycle:
      1. Open valve, retract plunger fully (fill syringe), close valve.
      2. Wait for user to press Enter.
      3. Dispense the target volume.
      4. Wait for user to measure and record.

    Results are saved incrementally to
    ``robot_control/data/<pump>_volume_cal_<timestamp>.csv`` and a
    corrected ``plunger_travel_mm`` is offered at the end.
    """
    print("\n" + "=" * 60)
    print("  CALIBRATION: VOLUME ACCURACY")
    print("=" * 60)

    pid = _select_single_pump(state, "Which pump to calibrate?")
    if not _check_homing(state, [pid]):
        return

    sy = state.syringe(pid)
    m = state.motor(pid)
    dsign = _dispense_sign(m)

    print(f"\n  {state.pump_label(pid)}")
    print(f"  Syringe: {sy.volume_ml} ml / {sy.plunger_travel_mm} mm "
          f"({sy.mm_per_ml:.4f} mm/ml)")

    # -- mode selection -----------------------------------------------------
    print("\n  Calibration modes:")
    print("    [1] Fixed volume -- repeat the same volume N times")
    print("    [2] Volume sweep -- 0.1 to 1.0 ml in 10 steps")
    mode = _input("Mode", "1")

    if mode == "2":
        volumes = [round(0.1 * i, 1) for i in range(1, 11)]
        volumes = [v for v in volumes if v <= sy.volume_ml]
        if not volumes:
            print("  !! Syringe capacity too small for sweep. Aborting.")
            return
        cycles = len(volumes)
        print(f"\n  Sweep: {volumes[0]:.1f} -> {volumes[-1]:.1f} ml "
              f"({cycles} steps)")
    else:
        target_vol = _input_float("Target volume (ml)", 0.5)
        if target_vol > sy.volume_ml:
            print(f"  !! Volume exceeds syringe capacity "
                  f"({sy.volume_ml:.2f} ml). Clamping.")
            target_vol = sy.volume_ml
        cycles = _input_int("Number of cycles", 5)
        volumes = [target_vol] * cycles

    speed = _input_float("Dispense speed (mm/s)", 0.5)

    # -- prepare CSV --------------------------------------------------------
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = data_dir / f"{pid}_volume_cal_{timestamp}.csv"
    _CSV_HEADER = (
        "cycle,target_ml,measured_ml,error_ml,error_pct,"
        "travel_mm,speed_mm_s,mm_per_ml_nominal,mm_per_ml_actual"
    )
    csv_lines: list[str] = [_CSV_HEADER]

    sock = state.ensure_connected()
    results: list[dict[str, float]] = []

    print(f"\n  Starting {cycles} calibration cycle(s)")
    print(f"  CSV: {csv_path}")
    print(f"  Speed: {speed} mm/s\n")

    try:
        for i, target in enumerate(volumes):
            cycle_num = i + 1
            travel_mm = volume_to_mm(target, sy)
            pos = dsign * travel_mm

            if not state.check_travel_limit(pid, pos):
                print("  Aborting calibration due to travel limit.")
                break

            print(f"  === Cycle {cycle_num}/{cycles}: "
                  f"target = {target:.3f} ml ({travel_mm:.4f} mm) ===")

            # Fill: open valve -> retract plunger to 0 -> close valve
            state.open_valve()
            try:
                pump_enable(sock, pid)
                pump_move(
                    sock, pid, position=0.0,
                    speed=m.max_retract_speed_mm_s,
                )
            finally:
                state.close_valve()

            input("\n  Ready to dispense. Press ENTER ...")

            # Dispense
            pump_set_position(sock, pid, 0.0)
            print(f"    Dispensing {target:.3f} ml ...", end="", flush=True)
            pump_move(sock, pid, position=pos, speed=speed)
            print(" done")

            input("  Measure the output. Press ENTER to record ...")

            measured = _input_float("Measured volume (ml)", target)
            if measured <= 0:
                print("  !! Must be > 0. Recording as target value.")
                measured = target

            error_ml = measured - target
            error_pct = (error_ml / target * 100.0) if target > 0 else 0.0
            actual_mm_per_ml = (
                travel_mm / measured if measured > 0 else 0.0
            )

            result: dict[str, float] = {
                "cycle": cycle_num,
                "target_ml": target,
                "measured_ml": measured,
                "error_ml": error_ml,
                "error_pct": error_pct,
                "travel_mm": travel_mm,
                "speed_mm_s": speed,
                "mm_per_ml_nominal": sy.mm_per_ml,
                "mm_per_ml_actual": actual_mm_per_ml,
            }
            results.append(result)

            csv_lines.append(
                f"{cycle_num},{target:.4f},{measured:.4f},{error_ml:.4f},"
                f"{error_pct:.2f},{travel_mm:.4f},{speed:.4f},"
                f"{sy.mm_per_ml:.4f},{actual_mm_per_ml:.4f}"
            )
            csv_path.write_text("\n".join(csv_lines) + "\n")

            print(f"    Error: {error_ml:+.4f} ml ({error_pct:+.2f}%)")
            print(f"    Actual mm/ml: {actual_mm_per_ml:.4f} "
                  f"(nominal {sy.mm_per_ml:.4f})\n")

        pump_disable(sock, pid)

    except KeyboardInterrupt:
        print("\n\n  Calibration interrupted.")
        pump_disable(sock, pid)
        if state.valve_open_flag:
            state.close_valve()

    if not results:
        print("  No data collected.")
        return

    # -- statistics ---------------------------------------------------------
    _print_cal_statistics(pid, state, results, csv_path)

    # -- correction ---------------------------------------------------------
    total_travel = sum(r["travel_mm"] for r in results)
    total_measured = sum(r["measured_ml"] for r in results)
    if total_measured > 0 and total_travel > 0:
        avg_mm_per_ml = total_travel / total_measured
        corrected_travel = avg_mm_per_ml * sy.volume_ml

        print(f"\n  Recommended plunger_travel_mm: {corrected_travel:.4f}")
        print(f"  (currently {sy.plunger_travel_mm:.4f})")

        if _input_bool("Save corrected value to machine.yaml?"):
            _save_volume_calibration({
                pid: {
                    "plunger_travel_mm": round(corrected_travel, 4),
                    "volume_ml": sy.volume_ml,
                },
            })
            print("  Saved. Restart the testbed to apply.")
        else:
            print("  Not saved.")


def _print_cal_statistics(
    pid: str,
    state: SessionState,
    results: list[dict[str, float]],
    csv_path: Path,
) -> None:
    """Print a formatted statistics summary for calibration results.

    Parameters
    ----------
    pid : str
        Pump identifier.
    state : SessionState
        Current session (used for pump label).
    results : list[dict[str, float]]
        Per-cycle measurement dicts produced by ``calibrate_volume``.
    csv_path : Path
        Path to the CSV file (shown in the summary header).
    """
    n = len(results)
    errors_ml = [r["error_ml"] for r in results]
    errors_pct = [r["error_pct"] for r in results]
    measured = [r["measured_ml"] for r in results]
    targets = [r["target_ml"] for r in results]
    actual_ratios = [r["mm_per_ml_actual"] for r in results]

    mean_err = sum(errors_ml) / n
    mean_err_pct = sum(errors_pct) / n
    max_abs_err = max(abs(e) for e in errors_ml)
    max_abs_err_pct = max(abs(e) for e in errors_pct)
    mean_mm_per_ml = sum(actual_ratios) / n

    # Population std dev (we have the full dataset, not a sample)
    var_ml = sum((e - mean_err) ** 2 for e in errors_ml) / n
    std_ml = var_ml ** 0.5
    var_pct = sum((e - mean_err_pct) ** 2 for e in errors_pct) / n
    std_pct = var_pct ** 0.5

    print("\n" + "=" * 60)
    print(f"  CALIBRATION RESULTS: {state.pump_label(pid)}")
    print("=" * 60)
    print(f"  Cycles completed: {n}")
    print(f"  CSV: {csv_path}")

    # Per-cycle table
    print(f"\n  {'#':>3s}  {'Target':>8s}  {'Measured':>9s}  "
          f"{'Error':>8s}  {'Error%':>7s}  {'mm/ml':>8s}")
    print(f"  {'---':>3s}  {'--------':>8s}  {'---------':>9s}  "
          f"{'--------':>8s}  {'-------':>7s}  {'--------':>8s}")
    for r in results:
        print(f"  {r['cycle']:3.0f}  {r['target_ml']:8.4f}  "
              f"{r['measured_ml']:9.4f}  {r['error_ml']:+8.4f}  "
              f"{r['error_pct']:+7.2f}  {r['mm_per_ml_actual']:8.4f}")

    # Aggregate stats
    print(f"\n  Summary:")
    print(f"    Mean error:     {mean_err:+.4f} ml  ({mean_err_pct:+.2f}%)")
    print(f"    Std dev:        {std_ml:.4f} ml  ({std_pct:.2f}%)")
    print(f"    Max |error|:    {max_abs_err:.4f} ml  "
          f"({max_abs_err_pct:.2f}%)")
    print(f"    Mean mm/ml:     {mean_mm_per_ml:.4f} "
          f"(nominal {results[0]['mm_per_ml_nominal']:.4f})")

    unique_targets = set(targets)
    if len(unique_targets) == 1:
        # Fixed-volume repeatability
        mean_m = sum(measured) / n
        var_rep = sum((m - mean_m) ** 2 for m in measured) / n
        std_rep = var_rep ** 0.5
        cv = (std_rep / mean_m * 100.0) if mean_m > 0 else 0.0
        print(f"\n  Repeatability (fixed {targets[0]:.3f} ml):")
        print(f"    Mean measured:  {mean_m:.4f} ml")
        print(f"    Std dev:        {std_rep:.4f} ml")
        print(f"    CV:             {cv:.2f}%")
    else:
        # Sweep linearity (R^2 between target and measured)
        mean_t = sum(targets) / n
        mean_m = sum(measured) / n
        ss_res = sum((m - t) ** 2 for m, t in zip(measured, targets))
        ss_tot = sum((m - mean_m) ** 2 for m in measured)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        print(f"\n  Linearity (volume sweep):")
        print(f"    R-squared:      {r_sq:.6f}")
        print(f"    Best-fit mm/ml: {mean_mm_per_ml:.4f}")


def _save_volume_calibration(
    corrections: dict[str, dict[str, float]],
) -> None:
    """Write corrected syringe params back to machine.yaml."""
    data = load_yaml(_CFG_PATH)

    for pid, corr in corrections.items():
        motors_data = data.get("pumps", {}).get("motors", {})
        if pid not in motors_data:
            print(f"  !! {pid} not found in machine.yaml motors section")
            continue
        motor_data = motors_data[pid]
        if "syringe" not in motor_data:
            motor_data["syringe"] = {}
        motor_data["syringe"]["plunger_travel_mm"] = corr["plunger_travel_mm"]
        motor_data["syringe"]["volume_ml"] = corr["volume_ml"]

    atomic_yaml_dump(data, _CFG_PATH)


def calibrate_mixing(state: SessionState) -> None:
    """Calibration 2: CMY mixing ratio -> RGB mapping.

    Dispenses a CMY ink ratio simultaneously, then records the measured
    RGB into ``manual_calibration_results.yaml``.
    """
    print("\n" + "=" * 60)
    print("  CALIBRATION: MIXING TEST")
    print("=" * 60)

    ink_ids = state.ink_pump_ids
    fluids = {pid: state.motor(pid).fluid for pid in ink_ids}

    print("\n  Ink pumps:")
    for pid in ink_ids:
        print(f"    {state.pump_label(pid)}")

    print("\n  Enter CMY ratio (must sum to 1.0, or will be normalized):")
    ratios: dict[str, float] = {}
    for pid in ink_ids:
        ratios[pid] = _input_float(
            f"  Ratio for {fluids[pid]}", round(1.0 / len(ink_ids), 2)
        )

    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        print("  !! Ratios must be positive. Aborting.")
        return
    if abs(ratio_sum - 1.0) > 0.001:
        print(f"  Normalizing ratios (sum was {ratio_sum:.3f})")
        ratios = {k: v / ratio_sum for k, v in ratios.items()}

    total_vol = _input_float("Total volume (ml)", 1.0)
    speed = _input_float("Speed (mm/s)", 0.5)

    if not _check_homing(state, ink_ids):
        return

    # Compute per-pump volumes
    pump_volumes = {pid: ratios[pid] * total_vol for pid in ink_ids}
    print("\n  Per-pump volumes:")
    for pid in ink_ids:
        print(f"    {state.pump_label(pid)}: "
              f"{pump_volumes[pid]:.3f} ml (ratio {ratios[pid]:.3f})")

    if not _input_bool("Proceed with dispensing?"):
        return

    _do_simultaneous_dispense(state, ink_ids, pump_volumes, speed, retract=False)
    for pid in ink_ids:
        pump_disable(state.ensure_connected(), pid)

    # Record result
    label = _input("Label for this mix (e.g. MIX_001)", "")
    if not label:
        label = f"MIX_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    cmy_ratio = [ratios[pid] for pid in ink_ids]

    print(f"\n  Label: {label}")
    print(f"  CMY ratio: {cmy_ratio}")
    print("  Apply the ink to paper, scan, and determine the RGB values.")
    print("  You can enter the RGB now or later in the YAML file.")

    enter_now = _input_bool("Enter measured RGB now?", default=False)
    rgb_measured = [0.0, 0.0, 0.0]
    if enter_now:
        rgb_measured[0] = _input_float("R (0-1)", 0.0)
        rgb_measured[1] = _input_float("G (0-1)", 0.0)
        rgb_measured[2] = _input_float("B (0-1)", 0.0)

    _save_mixing_result(label, cmy_ratio, total_vol, speed, rgb_measured)
    print(f"  Result saved to {_MANUAL_CAL_PATH}")

    # Offer purge
    if _input_bool("\n  Purge the manifold now?"):
        control_purge(state)


def _save_mixing_result(
    label: str,
    cmy_ratio: list[float],
    total_volume_ml: float,
    speed_mm_s: float,
    rgb_measured: list[float],
) -> None:
    """Append a mixing test result to manual_calibration_results.yaml."""
    data = load_yaml(_MANUAL_CAL_PATH)
    if data is None:
        data = {}

    if "ink_mixing_measurements" not in data:
        data["ink_mixing_measurements"] = {}

    data["ink_mixing_measurements"][label] = {
        "cmy_ratio": [round(r, 4) for r in cmy_ratio],
        "total_volume_ml": round(total_volume_ml, 4),
        "speed_mm_s": round(speed_mm_s, 4),
        "rgb_measured": [round(v, 6) for v in rgb_measured],
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": "",
    }

    atomic_yaml_dump(data, _MANUAL_CAL_PATH)


# ---------------------------------------------------------------------------
# Menu system
# ---------------------------------------------------------------------------

_TESTS = [
    ("1", "Pump juggle",            test_pump_juggle),
    ("2", "Volume test",            test_volume),
    ("3", "Limit switch test",      test_limit_switch),
    ("4", "Speed ramp",             test_speed_ramp),
    ("5", "Repeatability",          test_repeatability),
    ("6", "Full travel",            test_full_travel),
    ("7", "Volume repeatability",   test_volume_repeatability),
]

_CONTROLS = [
    ("1", "Homing",           control_homing),
    ("2", "Setup (backlash)", control_setup),
    ("3", "Individual pump",  control_individual_pump),
    ("4", "Multi-pump",       control_multi_pump),
    ("5", "Purge",            control_purge),
    ("6", "Fill",             control_fill),
    ("7", "Valve cycle",      control_valve_cycle),
]

_CALIBRATIONS = [
    ("1", "Volume accuracy",     calibrate_volume),
    ("2", "Mixing test",         calibrate_mixing),
]


def _run_submenu(
    title: str,
    items: list[tuple[str, str, Any]],
    state: SessionState,
) -> None:
    """Display a submenu and run the selected action in a loop."""
    while True:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)
        for key, label, _ in items:
            print(f"    [{key}] {label}")
        print(f"    [B] Back")

        choice = _input("Choice", "B").upper()
        if choice == "B":
            return

        handler = None
        for key, label, fn in items:
            if choice == key:
                handler = fn
                break

        if handler is None:
            print("  Invalid choice.")
            continue

        try:
            handler(state)
        except KeyboardInterrupt:
            print("\n\n  Interrupted.")
        except Exception as exc:
            print(f"\n  !! Error: {exc}")

        # After running, offer to repeat
        if not _input_bool("\n  Run another action from this menu?"):
            return


def _print_banner(state: SessionState) -> None:
    ps = state.pumps_cfg.stepper
    print()
    print("=" * 60)
    print("  PUMP TESTBED  (CMY + Purge, no XYZ motion)")
    print("=" * 60)
    print()
    print(f"  Pumps: {len(state.pump_ids)} configured")
    for pid in state.pump_ids:
        m = state.motor(pid)
        sy = state.syringe(pid)
        print(f"    {pid}: {m.fluid:<8s} {m.octopus_slot:<10s} "
              f"({sy.volume_ml} ml / {sy.plunger_travel_mm} mm)")
    print()
    print(f"  Stepper: {ps.motor_type}, "
          f"{ps.klipper_microsteps} microsteps, "
          f"{ps.rotation_distance} mm/rev")
    print(f"  Backlash purge: {state.pumps_cfg.backlash_purge_mm} mm")
    print(f"  Manifold purge: "
          f"{state.pumps_cfg.manifold_purge_volume_ml} ml")
    valve = state.pumps_cfg.refill_valve_output or "(none)"
    print(f"  Refill valve: {valve} "
          f"(delay {state.pumps_cfg.refill_valve_delay_s:.1f} s)")
    print(f"  Travel limits: "
          f"{'ENFORCED' if state.pumps_cfg.enforce_travel_limits else 'disabled'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive pump testbed (CMY + Purge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't regenerate printer.cfg",
    )
    args = parser.parse_args()

    cfg = load_config()
    if cfg.pumps is None:
        print("  !! Pumps are not enabled in machine.yaml")
        sys.exit(1)

    state = SessionState(cfg)

    # Write pump-only printer.cfg and restart Klipper
    if not args.no_config_write:
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            print(f"  Backed up printer.cfg -> {backup}")

        config_text = generate_pump_test_cfg(cfg, pump_ids=state.pump_ids)
        PRINTER_CFG_PATH.write_text(config_text)
        print(f"  Wrote pump test printer.cfg to {PRINTER_CFG_PATH}")
        print("  Restarting Klipper ...")
        restart_klipper(state.socket_path)

    print("  Waiting for Klipper ...")
    state.sock = wait_for_ready(state.socket_path, timeout=45.0)
    print("  [OK] Klipper is ready")

    # Ensure all pumps start disabled (no holding current)
    for pid in state.pump_ids:
        pump_disable(state.sock, pid)

    _print_banner(state)

    try:
        while True:
            print()
            print("=" * 60)
            print("  MAIN MENU")
            print("=" * 60)
            print("    [T] Tests")
            print("    [C] Control")
            print("    [L] Calibration")
            print("    [Q] Quit")

            homed_list = [pid for pid, h in state.homed.items() if h]
            if homed_list:
                print(f"\n  Homed: {', '.join(homed_list)}")

            choice = _input("Choice", "Q").upper()

            if choice == "T":
                _run_submenu("TESTS", _TESTS, state)
            elif choice == "C":
                _run_submenu("CONTROL", _CONTROLS, state)
            elif choice == "L":
                _run_submenu("CALIBRATION", _CALIBRATIONS, state)
            elif choice == "Q":
                break
            else:
                print("  Invalid choice.")

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        if state.sock is not None:
            # Ensure valve is closed on exit
            if state.valve_open_flag:
                try:
                    state.close_valve()
                except Exception:
                    pass
            # Disable all pump motors
            for pid in state.pump_ids:
                try:
                    pump_disable(state.sock, pid)
                except Exception:
                    pass
            state.sock.close()

    print("\n  Pump testbed session ended.")


if __name__ == "__main__":
    main()
