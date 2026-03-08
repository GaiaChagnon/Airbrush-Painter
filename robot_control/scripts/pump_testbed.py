#!/usr/bin/env python3
"""Interactive pump testbed for CMY + purge syringe pump system.

Menu-driven script for testing, controlling, and calibrating the four
syringe pumps (Cyan, Magenta, Yellow, Purge/IPA).  Axis motors are
disabled by default -- only pump steppers are active.

All machine parameters come from ``machine.yaml``; nothing is hardcoded.

Usage::

    python robot_control/scripts/pump_testbed.py
    python robot_control/scripts/pump_testbed.py --no-config-write
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
    """Tracks per-pump homing state and the active socket."""

    def __init__(self, cfg: MachineConfig) -> None:
        self.cfg = cfg
        self.pumps_cfg: PumpsConfig = cfg.pumps  # type: ignore[assignment]
        self.socket_path: str = cfg.connection.socket_path
        self.sock: socket.socket | None = None
        self.homed: dict[str, bool] = {
            pid: False for pid in self.pumps_cfg.motors
        }

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

    Motors are disabled after homing (both on success and failure).
    """
    sock = state.ensure_connected()
    backlash = state.pumps_cfg.backlash_purge_mm
    all_ok = True

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
        pump_enable(sock, pid)
        pump_set_position(sock, pid, 0.0)

        print(f"\n  Juggling {state.pump_label(pid)}: "
              f"{distance} mm x {reps} reps at {speed} mm/s")

        for i in range(1, reps + 1):
            pos = dsign * distance
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
    """Control 6: retract plunger(s) fully to fill syringes."""
    print("\n" + "=" * 60)
    print("  CONTROL: FILL (retract plunger)")
    print("=" * 60)

    selected = _select_pumps(state, "Which pumps to fill?")
    speed = _input_float("Retract speed (mm/s)", 2.0)
    if not _check_homing(state, selected):
        return

    sock = state.ensure_connected()
    for pid in selected:
        m = state.motor(pid)
        sy = state.syringe(pid)
        print(f"\n  Filling {state.pump_label(pid)} "
              f"({sy.volume_ml:.2f} ml) ...", end="", flush=True)
        pump_enable(sock, pid)
        pump_move(sock, pid, position=0.0, speed=speed)
        print(" done")
        pump_disable(sock, pid)

    print("\n  Fill complete.")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_volume(state: SessionState) -> None:
    """Calibration 1: volume accuracy per pump.

    Dispenses a target volume, user measures the actual output, and
    the script computes a corrected ``plunger_travel_mm`` so that
    ``mm_per_ml`` matches reality.
    """
    print("\n" + "=" * 60)
    print("  CALIBRATION: VOLUME")
    print("=" * 60)

    target_vol = _input_float("Target volume to dispense (ml)", 0.5)
    speed = _input_float("Speed (mm/s)", 0.5)

    selected = _select_pumps(
        state, "Which pumps to calibrate?", allow_all=True,
    )
    if not _check_homing(state, selected):
        return

    corrections: dict[str, dict[str, float]] = {}

    for pid in selected:
        sy = state.syringe(pid)
        print(f"\n  --- {state.pump_label(pid)} ---")
        print(f"  Current: {sy.volume_ml} ml / {sy.plunger_travel_mm} mm "
              f"({sy.mm_per_ml:.2f} mm/ml)")

        _do_dispense_no_retract(state, pid, target_vol, speed)

        if not _input_bool("Enter measured volume?", default=True):
            print("  Skipped measurement.")
            continue

        measured = _input_float(
            "Measured volume output (ml)", target_vol
        )
        if measured <= 0:
            print("  !! Measured volume must be > 0. Skipping.")
            continue

        travel_used = volume_to_mm(target_vol, sy)
        actual_mm_per_ml = travel_used / measured
        corrected_travel = actual_mm_per_ml * sy.volume_ml

        print(f"  Nominal mm/ml:   {sy.mm_per_ml:.4f}")
        print(f"  Actual mm/ml:    {actual_mm_per_ml:.4f}")
        print(f"  Corrected plunger_travel_mm: {corrected_travel:.4f}")

        corrections[pid] = {
            "plunger_travel_mm": round(corrected_travel, 4),
            "volume_ml": sy.volume_ml,
        }

    if corrections:
        print("\n  Summary of corrections:")
        for pid, corr in corrections.items():
            print(f"    {pid}: plunger_travel_mm = {corr['plunger_travel_mm']}")

        if _input_bool("Save corrected values to machine.yaml?"):
            _save_volume_calibration(corrections)
            print("  Saved. Reload config to apply.")
        else:
            print("  Not saved.")


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
]

_CALIBRATIONS = [
    ("1", "Volume calibration",  calibrate_volume),
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
        # Disable all pump motors
        if state.sock is not None:
            for pid in state.pump_ids:
                try:
                    pump_disable(state.sock, pid)
                except Exception:
                    pass
            state.sock.close()

    print("\n  Pump testbed session ended.")


if __name__ == "__main__":
    main()
