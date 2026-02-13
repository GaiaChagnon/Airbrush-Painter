#!/usr/bin/env python3
"""Hardware bring-up: motor spin, endstop verification, homing, and circle test.

Writes a ``printer.cfg`` using ``kinematics: cartesian`` so Klipper manages
coordinated motion natively.  The dual-X axis uses ``[stepper_x]`` +
``[stepper_x1]`` -- Klipper's native multi-stepper rail mechanism.
Both motors are on the same kinematic rail, so they move in perfect
hardware-level sync for ALL operations (G1, G28, FORCE_MOVE).

Phases
------
    Phase 1 -- Motor 0 (Y = stepper_y):     FORCE_MOVE spin test
    Phase 2 -- Motor 1 (dummy Z = stepper_z): FORCE_MOVE spin test
    Phase 3 -- Motor 2_1 + 2_2 (X dual):    G1 spin test (synced)
    Phase 4 -- Endstop verification:         QUERY_ENDSTOPS polling
    Phase 5 -- Homing:                       G28 X Y (both X motors stop
                                             together at endstop)
    Phase 6 -- Circle test:                  G1 XY moves within workspace

Motor parameters
----------------
    0.9deg motor, DM542TE driver (DIP: 6400 pulses/rev = 16 microsteps)
    rotation_distance=32 mm (GT2 2mm pitch x 16T pulley)
    step_pulse_duration=0.000005 (5 us) -- MANDATORY for DM542TE via LS08

Usage::

    python robot_control/scripts/test_motors.py
    python robot_control/scripts/test_motors.py --skip-motor1
    python robot_control/scripts/test_motors.py --endstops-only
    python robot_control/scripts/test_motors.py --speed 60
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOCKET_PATH = "/tmp/klippy_uds"
PRINTER_CFG_PATH = Path.home() / "printer.cfg"
ETX = b"\x03"

# Motor parameters -- must match physical DM542TE DIP switch setting.
#
# The DM542TE driver is configured for 6400 pulses/rev (= 16 microsteps
# on a 0.9deg / 400-step motor).  Klipper's microsteps MUST match the
# driver's setting so that 1 Klipper step = 1 driver pulse.
#
# With microsteps=16 and rotation_distance=32:
#   steps_per_mm = (400 * 16) / 32 = 200
#   MOVE=32 -> 32 * 200 = 6400 pulses = exactly 1 revolution
#
# Previous bug: microsteps=4 sent only 1600 pulses = quarter turn.
ROTATION_DISTANCE = 32.0   # mm per revolution: GT2 2mm pitch x 16T pulley
MICROSTEPS = 16             # must match DM542TE DIP setting (6400 pulses/rev)
DIRECTION_PAUSE_MS = 500    # 500 ms between rapid direction reversals

# step_pulse_duration MUST be 0.000005 (5 us) for DM542TE via LS08 buffer.
# Without this, the driver misses steps and motors don't complete full rotations.
# Written as a string to avoid Python's scientific notation (5e-06) in the config.
STEP_PULSE_DURATION = "0.000005"

# Pin assignments -- Octopus Pro V1.0.1 H723
#
# enable_pin: No "!" prefix needed.  Confirmed working without inversion
# in all prior test runs (motors spin correctly).  Adding "!" inverts the
# logic and prevents motors from moving entirely.
#
# motor_x2 dir_pin is inverted (!) because on a dual-motor gantry the two
# X motors face each other -- inverting one makes them spin the same physical
# direction so the carriage moves correctly.
#
# Endstop pins: bare pin name; prefix (^ or ^!) added in generate_test_config.
# Default is ^PIN (pull-up, no invert): switches connect to GND when triggered.
MOTOR_PINS = {
    "motor_y": {  # Motor 0 -- Y axis
        "step": "PF13", "dir": "PF12", "enable": "PF14",
        "endstop_pin": "PG9",   # STOP_1 -- Y endstop (was PG6, swapped)
    },
    "motor_1": {  # Motor 1 -- DRIVER_1 slot (unassigned axis)
        "step": "PG0", "dir": "PG1", "enable": "PF15",
        "endstop_pin": None,
    },
    "motor_x1": {  # Motor 2_1 -- X primary (endstop owner for the X gantry)
        "step": "PF11", "dir": "PG3", "enable": "PG5",
        "endstop_pin": "PG6",   # STOP_0 -- X endstop (was PG9, swapped)
    },
    "motor_x2": {  # Motor 2_2 -- X secondary (dir inverted, ALWAYS moves with x1)
        "step": "PG4", "dir": "!PC1", "enable": "PA0",
        "endstop_pin": None,    # No own endstop -- shares X endstop via motor_x1
    },
}

# The two X motors (motor_x1, motor_x2) are on the same gantry bar.
# They MUST always be enabled, moved, and disabled together.
X_MOTORS = ("motor_x1", "motor_x2")

# Workspace bounds (mm) -- defines position_max in cartesian stepper config.
WORKSPACE_X_MM = 200.0
WORKSPACE_Y_MM = 200.0

# Homing speed: 0.5 RPS = 16 mm/s with 32 mm rotation_distance
HOMING_SPEED = 16.0


# ---------------------------------------------------------------------------
# Printer.cfg generation
# ---------------------------------------------------------------------------


def generate_test_config(invert_endstops: bool = True) -> str:
    """Generate a printer.cfg using ``kinematics: cartesian``.

    This gives us proper coordinated motion and -- critically --
    hardware-level dual-motor homing via ``SYNC_TO``.

    Layout
    ------
    [stepper_x]   = motor_x1 (primary X, owns the X endstop)
    [stepper_x1]  = motor_x2 (extra stepper on X rail, auto-synced)
    [stepper_y]   = motor_y  (Y axis, owns the Y endstop)
    [stepper_z]   = motor_1  (dummy Z, never homed physically)

    Klipper's ``LookupMultiRail`` automatically finds ``[stepper_x1]``
    and adds it to the same rail as ``[stepper_x]``.  Both motors move
    together for all operations -- no sync commands needed.

    Endstop pins use ``^!PIN`` by default (pull-up + inverted).
    """
    inv = "!" if invert_endstops else ""

    # Common stepper params shared by all motors
    common = (
        f"microsteps: {MICROSTEPS}\n"
        f"full_steps_per_rotation: 400\n"  # 0.9-degree motor
        f"rotation_distance: {ROTATION_DISTANCE}\n"
        f"step_pulse_duration: {STEP_PULSE_DURATION}"
    )

    cfg = f"""\
# === AUTO-GENERATED TEST CONFIG -- DO NOT USE FOR PRODUCTION ===
# Generated by robot_control/scripts/test_motors.py
#
# Cartesian kinematics with dual-X via [stepper_x1] (extra stepper on X rail).
# Klipper automatically moves stepper_x and stepper_x1 together for ALL
# operations: G1, G28, FORCE_MOVE, etc.  No sync commands needed.
#
# 0.9deg motor, DM542TE, {MICROSTEPS} microsteps, {ROTATION_DISTANCE} mm/rev
# step_pulse_duration {STEP_PULSE_DURATION} (5 us) -- MANDATORY for DM542TE/LS08
# endstop polarity: ^{inv}PIN

[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32h723xx_130028001051313234353230-if00

[printer]
kinematics: cartesian
max_velocity: 200
max_accel: 1500
max_z_velocity: 25
max_z_accel: 100
square_corner_velocity: 8.0

[force_move]
enable_force_move: True

[gcode_arcs]
resolution: 0.5

# --- X axis primary (motor_x1) ---
[stepper_x]
step_pin: {MOTOR_PINS['motor_x1']['step']}
dir_pin: {MOTOR_PINS['motor_x1']['dir']}
enable_pin: {MOTOR_PINS['motor_x1']['enable']}
{common}
endstop_pin: ^{inv}{MOTOR_PINS['motor_x1']['endstop_pin']}
position_endstop: 0
position_min: 0
position_max: {WORKSPACE_X_MM}
homing_speed: {HOMING_SPEED}

# --- X axis secondary (motor_x2) -- auto-synced to stepper_x rail ---
[stepper_x1]
step_pin: {MOTOR_PINS['motor_x2']['step']}
dir_pin: {MOTOR_PINS['motor_x2']['dir']}
enable_pin: {MOTOR_PINS['motor_x2']['enable']}
{common}

# --- Y axis ---
[stepper_y]
step_pin: {MOTOR_PINS['motor_y']['step']}
dir_pin: {MOTOR_PINS['motor_y']['dir']}
enable_pin: {MOTOR_PINS['motor_y']['enable']}
{common}
endstop_pin: ^{inv}{MOTOR_PINS['motor_y']['endstop_pin']}
position_endstop: {WORKSPACE_Y_MM}
position_min: 0
position_max: {WORKSPACE_Y_MM}
homing_speed: {HOMING_SPEED}
homing_positive_dir: True

# --- Z axis (dummy -- motor_1 pins, never physically homed) ---
[stepper_z]
step_pin: {MOTOR_PINS['motor_1']['step']}
dir_pin: {MOTOR_PINS['motor_1']['dir']}
enable_pin: {MOTOR_PINS['motor_1']['enable']}
{common}
endstop_pin: ^{inv}PG10
position_endstop: 0
position_min: -1
position_max: 1
homing_positive_dir: False
"""
    return cfg


# ---------------------------------------------------------------------------
# Low-level Klipper comms (raw socket)
# ---------------------------------------------------------------------------


# Monotonically increasing request ID so we can match responses to requests
# even when async notifications arrive interleaved on the socket.
_next_id = 1


def _raw_send(sock: socket.socket, method: str, params: dict, timeout: float = 10.0) -> dict:
    """Send one JSON request and return the matching response dict.

    Uses an incrementing request ``id`` and skips any interleaved
    Klipper notifications (which lack an ``id`` field) until the
    response with the correct ``id`` arrives.
    """
    global _next_id
    req_id = _next_id
    _next_id += 1

    payload = json.dumps({"id": req_id, "method": method, "params": params}).encode() + ETX
    sock.sendall(payload)
    sock.settimeout(timeout)

    buf = b""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Accumulate data until we have at least one ETX-terminated frame
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

        # Process all complete frames in buffer
        while ETX in buf:
            idx = buf.index(ETX)
            frame = buf[:idx]
            buf = buf[idx + 1:]
            try:
                msg = json.loads(frame.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Match: response has our id (Klipper notifications have no id)
            if msg.get("id") == req_id:
                return msg

        if ETX not in buf and time.monotonic() >= deadline:
            break

    # Timeout -- return empty dict so callers don't crash
    return {}


def _drain_socket(sock: socket.socket, duration: float = 0.2) -> None:
    """Read and discard any pending data in the socket buffer.

    Call this before critical command sequences to clear stale
    notifications and out-of-order responses that would otherwise
    be consumed by the next ``_raw_send`` call.
    """
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        try:
            data = sock.recv(4096)
            if not data:
                break
        except socket.timeout:
            break


def _raw_gcode(sock: socket.socket, script: str, timeout: float = 30.0) -> bool:
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
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            resp = _raw_send(sock, "info", {})
            result = resp.get("result", {})
            state = result.get("state", "unknown")
            if state == "ready":
                return sock
            sock.close()
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Klipper did not become ready within {timeout}s")


def restart_klipper() -> None:
    """Send RESTART via the API and wait for Klipper to come back."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(SOCKET_PATH)
        _raw_gcode(sock, "RESTART")
        sock.close()
    except OSError:
        pass
    time.sleep(3.0)


# ---------------------------------------------------------------------------
# Motor test helpers
# ---------------------------------------------------------------------------


def spin_single_stepper(
    sock: socket.socket,
    stepper_name: str,
    cycles: int = 3,
    speed: float = 40.0,
) -> None:
    """Spin a cartesian stepper back and forth using FORCE_MOVE.

    FORCE_MOVE moves a stepper by a relative distance without requiring
    homing.  Each cycle: forward 1 revolution (32 mm), pause, reverse.

    Parameters
    ----------
    stepper_name : str
        Klipper stepper name, e.g. ``stepper_y``, ``stepper_z``.
    """
    one_rev = ROTATION_DISTANCE

    for i in range(1, cycles + 1):
        print(f"    Cycle {i}/{cycles}: forward 360deg ...", end="", flush=True)
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE={one_rev} VELOCITY={speed} ACCEL=4000",
        )
        print(" done")

        _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")

        print(f"    Cycle {i}/{cycles}: reverse 360deg ...", end="", flush=True)
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE=-{one_rev} VELOCITY={speed} ACCEL=4000",
        )
        print(" done")

        if i < cycles:
            _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")


def spin_x_back_and_forth(
    sock: socket.socket,
    cycles: int = 3,
    speed: float = 40.0,
) -> None:
    """Spin both X motors together using SET_KINEMATIC_POSITION + G1.

    motor_x2 must already be SYNC_TO=stepper_x.  Moves X +-32 mm
    (1 revolution) so both motors spin in perfect sync.
    """
    one_rev = ROTATION_DISTANCE
    feedrate = speed * 60.0  # mm/s -> mm/min

    for i in range(1, cycles + 1):
        # Start from known position 0
        _raw_gcode(sock, "SET_KINEMATIC_POSITION X=0 Y=0 Z=0")

        _drain_socket(sock)
        print(f"    Cycle {i}/{cycles}: forward 360deg (both) ...", end="", flush=True)
        _raw_gcode(sock, f"G1 X{one_rev:.1f} F{feedrate:.0f}")
        print(" done")

        _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")
        _drain_socket(sock)

        print(f"    Cycle {i}/{cycles}: reverse 360deg (both) ...", end="", flush=True)
        _raw_gcode(sock, f"G1 X0 F{feedrate:.0f}")
        print(" done")

        if i < cycles:
            _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")


# ---------------------------------------------------------------------------
# Endstop test -- live polling via objects/query
# ---------------------------------------------------------------------------

# Previous approach used gcode/subscribe_output to capture QUERY_ENDSTOPS
# terminal output.  That was fragile (message ordering, notification loss).
#
# New approach: send QUERY_ENDSTOPS via gcode/script, then immediately
# query the Klipper object model for the cached result:
#   objects/query {"objects": {"query_endstops": ["last_query"]}}
# This returns a structured dict, no text parsing needed.
#
# For manual_stepper endstops, the keys look like:
#   "manual_stepper motor_y": "open"
#   "manual_stepper motor_x1": "TRIGGERED"


def query_endstops(sock: socket.socket, debug: bool = False) -> dict[str, str]:
    """Query all endstop states via the Klipper object model.

    Sends QUERY_ENDSTOPS (to refresh the cached state), waits for
    Klipper to actually sample the pins, then reads the cached result
    from ``query_endstops.last_query``.

    Parameters
    ----------
    debug : bool
        If True, print the raw API response for troubleshooting.

    Returns
    -------
    dict[str, str]
        Mapping of short motor name to state string, e.g.
        ``{"motor_y": "open", "motor_x1": "TRIGGERED"}``.
    """
    # Drain any stale messages so the next reads are fresh
    _drain_socket(sock)

    # Step 1: Run QUERY_ENDSTOPS to refresh the cached state
    _raw_gcode(sock, "QUERY_ENDSTOPS")

    # Give Klipper time to actually sample the pins and update the cache.
    # Without this, the objects/query response can return stale data from
    # the previous QUERY_ENDSTOPS or from boot.
    time.sleep(0.3)

    # Drain the QUERY_ENDSTOPS text notification that Klipper sends
    _drain_socket(sock)

    # Step 2: Read the cached result from the object model
    resp = _raw_send(
        sock,
        "objects/query",
        {"objects": {"query_endstops": ["last_query"]}},
        timeout=3.0,
    )

    if debug:
        print(f"\n  [DEBUG] objects/query raw response: {json.dumps(resp, indent=2)}")

    results: dict[str, str] = {}
    status = resp.get("result", {}).get("status", {})
    last_query = status.get("query_endstops", {}).get("last_query", {})

    if debug and not last_query:
        print("  [DEBUG] last_query is empty -- objects/query may have returned "
              "a stale notification instead of the actual response.")

    # With cartesian kinematics, keys are "x", "y", "z" (not
    # "manual_stepper motor_y").  Normalise to consistent names.
    for full_name, raw_state in last_query.items():
        # Normalise: Klipper may return int (0/1) or string
        if isinstance(raw_state, int):
            state = "TRIGGERED" if raw_state else "open"
        else:
            state = str(raw_state)

        results[full_name] = state

    return results


def _stdin_has_data() -> bool:
    """Check if stdin has data ready (non-blocking)."""
    try:
        return bool(select.select([sys.stdin], [], [], 0.0)[0])
    except Exception:
        return False


def diagnose_mcu(sock: socket.socket) -> None:
    """Query and print MCU build info to verify firmware matches hardware.

    Queries the Klipper object model for MCU version, build config,
    and pin capabilities.  This helps diagnose whether the firmware
    was compiled for the correct MCU (STM32H723) and whether GPIO
    pins are accessible.
    """
    print("  --- MCU Firmware Diagnostic ---")

    # Query MCU object for version/build info
    _drain_socket(sock)
    try:
        resp = _raw_send(
            sock,
            "objects/query",
            {"objects": {"mcu": None}},
            timeout=5.0,
        )
        mcu_info = resp.get("result", {}).get("status", {}).get("mcu", {})

        version = mcu_info.get("mcu_version", "unknown")
        build = mcu_info.get("mcu_build_versions", "unknown")
        constants = mcu_info.get("mcu_constants", {})
        mcu_name = constants.get("MCU", "unknown")
        freq = constants.get("CLOCK_FREQ", 0)

        print(f"    MCU type:      {mcu_name}")
        print(f"    Clock freq:    {freq:,} Hz" if freq else "    Clock freq:    unknown")
        print(f"    FW version:    {version}")
        print(f"    Build:         {str(build)[:80]}")

        # Check if it's actually H723
        if "stm32h723" in str(mcu_name).lower():
            print("    [OK] Firmware is built for STM32H723")
        elif "stm32" in str(mcu_name).lower():
            print(f"    [!!] Firmware is for {mcu_name}, NOT H723!")
            print("         This could cause wrong pin mappings.")
        else:
            print(f"    [??] Could not confirm MCU type: {mcu_name}")

    except Exception as exc:
        print(f"    [!!] Failed to query MCU info: {exc}")

    # Query configfile to see loaded endstop pins
    _drain_socket(sock)
    try:
        resp = _raw_send(
            sock,
            "objects/query",
            {"objects": {"configfile": ["config"]}},
            timeout=5.0,
        )
        config = resp.get("result", {}).get("status", {}).get("configfile", {}).get("config", {})

        print()
        print("    Loaded endstop pins from config:")
        for section_name, section_data in config.items():
            if section_name.startswith(("stepper_", "manual_stepper")):
                ep = section_data.get("endstop_pin", None)
                if ep:
                    print(f"      [{section_name}] endstop_pin = {ep}")

    except Exception as exc:
        print(f"    [!!] Failed to query config: {exc}")

    print()


def run_endstop_test(sock: socket.socket) -> bool:
    """Quick endstop verification via live polling.

    For each endstop, polls QUERY_ENDSTOPS every 0.3s.  Auto-advances
    to the next endstop as soon as an OPEN -> TRIGGERED transition is
    detected.  Press Enter to skip if an endstop is stuck.

    Returns True if all endstops passed.
    """
    print()
    print("=" * 60)
    print("  PHASE 4: ENDSTOP VERIFICATION")
    print("=" * 60)
    print()
    print("  Press each endstop when prompted.  The test auto-advances")
    print("  once it detects the OPEN -> TRIGGERED transition.")
    print("  Press Enter to skip a stuck endstop.")
    print()

    # Discover endstop key names from Klipper (varies by kinematics).
    # cartesian: "x", "y", "z"  or  "stepper_x", "stepper_y", "stepper_z"
    # manual_stepper: "manual_stepper motor_y", etc.
    try:
        initial = query_endstops(sock)
    except Exception as exc:
        print(f"  !! Could not query endstops: {exc}")
        return False

    print(f"  Detected endstop keys: {list(initial.keys())}")
    print()

    # Find the key that corresponds to X and Y endstops
    def _find_key(states: dict[str, str], axis: str) -> str | None:
        """Find the endstop key for a given axis (x or y)."""
        # Try exact match first, then prefix match
        for candidate in [axis, f"stepper_{axis}", f"manual_stepper motor_{axis}",
                          f"manual_stepper motor_{axis}1"]:
            if candidate in states:
                return candidate
        # Fallback: any key containing the axis letter
        for key in states:
            if axis in key.lower():
                return key
        return None

    x_key = _find_key(initial, "x")
    y_key = _find_key(initial, "y")

    if not x_key:
        print(f"  !! No X endstop key found in: {list(initial.keys())}")
        return False
    if not y_key:
        print(f"  !! No Y endstop key found in: {list(initial.keys())}")
        return False

    endstops_to_test = [
        (x_key, "X axis endstop (STOP_0 / PG6)"),
        (y_key, "Y axis endstop (STOP_1 / PG9)"),
    ]

    all_pass = True

    for endstop_name, description in endstops_to_test:
        print(f"  >> {description}  [key: {endstop_name}]")
        print(f"     Release the switch, then press it.")

        saw_open = False
        saw_triggered = False

        while True:
            try:
                states = query_endstops(sock)
            except Exception as exc:
                print(f"\r     !! Query error: {exc}                    ")
                time.sleep(1.0)
                continue

            state = states.get(endstop_name, "???")

            if state.lower() == "open":
                saw_open = True
                indicator = "[ OPEN     ]"
            elif state.upper() == "TRIGGERED":
                saw_triggered = True
                indicator = "[TRIGGERED!]"
            else:
                indicator = f"[{state:^11s}]"

            sys.stdout.write(f"\r     {endstop_name}: {indicator}    ")
            sys.stdout.flush()

            # Auto-advance: saw both states (open then triggered)
            if saw_open and saw_triggered:
                print(f"\n     PASS  {endstop_name}")
                break

            # Allow Enter to skip
            if _stdin_has_data():
                sys.stdin.readline()
                print()
                if saw_open and not saw_triggered:
                    print(f"     FAIL  {endstop_name}: saw OPEN but never TRIGGERED")
                elif saw_triggered and not saw_open:
                    print(f"     FAIL  {endstop_name}: always TRIGGERED, never OPEN")
                else:
                    print(f"     SKIP  {endstop_name}")
                all_pass = False
                break

            time.sleep(0.3)

        print()

    tag = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"  ENDSTOP RESULT: {tag}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Homing & coordinated motion (cartesian kinematics + SYNC_TO)
# ---------------------------------------------------------------------------


def home_all(sock: socket.socket) -> bool:
    """Home X and Y using G28.  Both X motors move together.

    stepper_x and stepper_x1 are on the same Klipper rail, so G28
    moves both and both stop when the endstop triggers.
    Z is faked with SET_KINEMATIC_POSITION (no physical Z axis).

    After homing:
      X = 0   (endstop at min,  workspace 0..200)
      Y = 200 (endstop at max,  workspace 0..200)
    """
    print()
    print("=" * 60)
    print("  PHASE 5: HOMING  (G28 X Y, ~0.5 RPS)")
    print("=" * 60)
    print()

    # Fake Z position so G1 commands work later (no physical Z axis)
    _raw_gcode(sock, "SET_KINEMATIC_POSITION Z=0")

    # Home X and Y -- motor_x2 follows stepper_x via SYNC_TO
    print("  Homing X (both motors) and Y ...")
    ok = _raw_gcode(sock, "G28 X Y", timeout=60.0)
    if ok:
        print(f"    Homed OK  X=0  Y={WORKSPACE_Y_MM}")
        print(f"    Workspace: 0..{WORKSPACE_X_MM} mm (X)  x  "
              f"0..{WORKSPACE_Y_MM} mm (Y)")
    else:
        print("    Homing FAILED")
    print()
    return ok


def move_xy(
    sock: socket.socket,
    x: float,
    y: float,
    feedrate: float = 1200.0,
) -> None:
    """Move to (x, y) using G1.  Both X motors move in sync.

    Parameters
    ----------
    feedrate : float
        Speed in mm/min (G-code F parameter).  1200 = 20 mm/s.
    """
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{feedrate:.0f}")


def draw_circle(
    sock: socket.socket,
    center_x: float,
    center_y: float,
    radius: float,
    feedrate: float = 960.0,
) -> None:
    """Draw a circle using G2 arc commands.

    Uses two G2 semicircles (universally supported by G-code
    interpreters).  Klipper's ``[gcode_arcs]`` module internally
    interpolates each arc into smooth micro-segments with seamless
    velocity blending -- no deceleration between segments, so the
    motion is silent and vibration-free at any speed.

    Parameters
    ----------
    center_x, center_y : float
        Circle centre in mm.
    radius : float
        Circle radius in mm.
    feedrate : float
        Travel speed in mm/min (960 = 16 mm/s).
    """
    speed_mmps = feedrate / 60.0
    print(f"  Drawing circle: centre=({center_x},{center_y}) "
          f"r={radius} mm  {speed_mmps:.0f} mm/s  (G2 arcs)")

    # Move to start point (3 o'clock position)
    start_x = center_x + radius
    start_y = center_y
    move_xy(sock, start_x, start_y, feedrate)

    # Two semicircles: 3 o'clock -> 9 o'clock -> 3 o'clock
    # I, J = offset from current position to arc centre.
    #
    # First half: (cx+r, cy) -> (cx-r, cy)
    #   centre offset: I = cx - (cx+r) = -r,  J = cy - cy = 0
    _raw_gcode(
        sock,
        f"G2 X{center_x - radius:.2f} Y{center_y:.2f} "
        f"I{-radius:.2f} J0 F{feedrate:.0f}",
    )

    # Second half: (cx-r, cy) -> (cx+r, cy)
    #   centre offset: I = cx - (cx-r) = +r,  J = cy - cy = 0
    _raw_gcode(
        sock,
        f"G2 X{center_x + radius:.2f} Y{center_y:.2f} "
        f"I{radius:.2f} J0 F{feedrate:.0f}",
    )

    print(f"    circle done")


def run_motion_test(sock: socket.socket) -> None:
    """Phase 6: slow circle, then progressive speed ramp.

    1. Slow verification circle at 16 mm/s  (sanity check).
    2. Speed ramp: 30, 50, 80, 120, 160 mm/s circles (same 40 mm radius).
       Each is a single full circle -- short enough to abort quickly
       if something sounds wrong (Ctrl-C).

    All speeds are conservative for a belt-driven gantry with DM542TE
    drivers (max_velocity in Klipper config is 300 mm/s).
    """
    print()
    print("=" * 60)
    print("  PHASE 6: CIRCLE + SPEED RAMP TEST")
    print("=" * 60)
    print()

    cx = WORKSPACE_X_MM / 2.0   # 100 mm
    cy = WORKSPACE_Y_MM / 2.0   # 100 mm
    radius = 40.0               # 40 mm radius, well within bounds

    # Move to centre
    print(f"  Moving to workspace centre ({cx}, {cy}) ...")
    move_xy(sock, cx, cy, feedrate=1800.0)
    print("    Arrived.")
    print()

    # --- Slow verification circle ---
    print("  --- Slow circle (16 mm/s) ---")
    draw_circle(sock, cx, cy, radius, feedrate=960.0)
    print()

    # --- Speed ramp ---
    # G2 arcs produce smooth motion at any speed (no segment junctions).
    ramp_speeds = [50, 100, 160, 220, 250, 280, 310]  # mm/s
    print(f"  --- Speed ramp: {ramp_speeds} mm/s  (G2 arcs) ---")
    print(f"  (Ctrl-C to abort if motion becomes rough)")
    print()

    for speed in ramp_speeds:
        feedrate = speed * 60.0
        print(f"  --- {speed} mm/s ---")
        draw_circle(sock, cx, cy, radius, feedrate=feedrate)
        print()

    # Return to centre
    print("  Returning to centre ...")
    move_xy(sock, cx, cy, feedrate=1800.0)
    print("    Done.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hardware bring-up: motor spin + endstop test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-motor1", action="store_true",
        help="Skip Motor 1 (DRIVER_1) test if nothing is connected there",
    )
    parser.add_argument(
        "--endstops-only", action="store_true",
        help="Skip motor tests, run endstop verification only",
    )
    parser.add_argument(
        "--skip-to-homing", action="store_true",
        help="Skip motor spin tests and endstop checks, go straight to homing + circle",
    )
    parser.add_argument(
        "--speed", type=float, default=40.0,
        help="Test speed in mm/s (default: 40 = ~1.25 RPS)",
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of back-and-forth cycles per motor (default: 3)",
    )
    parser.add_argument(
        "--no-config-write", action="store_true",
        help="Don't overwrite printer.cfg (assume correct config is loaded)",
    )
    parser.add_argument(
        "--no-invert-endstops", action="store_true",
        help="Use ^PIN (pull-up only, no inversion) instead of the default "
             "^!PIN. Default is ^!PIN (pull-up + inverted) which is correct "
             "when the endstop reads HIGH when open and LOW when pressed.",
    )
    args = parser.parse_args()

    invert_endstops = not args.no_invert_endstops
    rps = args.speed / ROTATION_DISTANCE

    print("=" * 60)
    print("  HARDWARE BRING-UP TEST")
    print("=" * 60)
    print()
    print("  Motor parameters:")
    print(f"    rotation_distance:  {ROTATION_DISTANCE} mm/rev")
    print(f"    microsteps:         {MICROSTEPS}  (must match DM542TE DIP)")
    print(f"    step_pulse:         {STEP_PULSE_DURATION} s (5 us)")
    print(f"    pulses/rev:         {400 * MICROSTEPS}  (400 full-steps x {MICROSTEPS})")
    print(f"    test speed:         {args.speed} mm/s ({rps:.2f} RPS)")
    print(f"    direction pause:    {DIRECTION_PAUSE_MS} ms")
    print(f"    cycles:             {args.cycles} back-and-forth")
    endstop_mode = "^!PIN (inverted)" if invert_endstops else "^PIN (normal)"
    print(f"    endstop polarity:   {endstop_mode}")
    print()

    # --- Write test config and restart Klipper ----------------------------
    if not args.no_config_write:
        # Back up existing config
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            print(f"  Backed up existing printer.cfg -> {backup}")

        config_text = generate_test_config(invert_endstops=invert_endstops)
        PRINTER_CFG_PATH.write_text(config_text)
        print(f"  Wrote test printer.cfg to {PRINTER_CFG_PATH}")
        print()

        print("  Restarting Klipper to load test config...")
        restart_klipper()

    print("  Waiting for Klipper to become ready...")
    sock = _wait_for_ready(timeout=30.0)
    print("  [OK] Klipper is ready")
    print()

    # --- MCU diagnostic: verify firmware matches the board -----------------
    diagnose_mcu(sock)

    try:
        if not args.skip_to_homing:
            # --- Phase 1: Motor 0 (Y) ------------------------------------
            if not args.endstops_only:
                print("=" * 60)
                print("  PHASE 1: Motor 0 (Y axis = stepper_y)")
                print("  Pins: step=PF13  dir=PF12  enable=PF14")
                print("  Uses FORCE_MOVE for individual stepper test")
                print("=" * 60)
                print()

                spin_single_stepper(sock, "stepper_y", args.cycles, args.speed)
                print()
                print("  Motor 0 test complete.")
                print()

                # --- Phase 2: Motor 1 (dummy Z = stepper_z) --------------
                if not args.skip_motor1:
                    print()
                    print("=" * 60)
                    print("  PHASE 2: Motor 1 (DRIVER_1 = stepper_z dummy)")
                    print("  Pins: step=PG0  dir=PG1  enable=PF15")
                    print("=" * 60)
                    print()
                    print("  NOTE: This motor slot is mapped to the dummy Z axis.")
                    print("  If nothing is connected, use --skip-motor1 to skip.")
                    print()

                    spin_single_stepper(sock, "stepper_z", args.cycles, args.speed)
                    print()
                    print("  Motor 1 test complete.")
                    print()
                else:
                    print()
                    print("  [SKIP] Motor 1 (--skip-motor1)")
                    print()

                # --- Phase 3: Motor 2_1 + 2_2 (X dual, same rail) --------
                print()
                print("=" * 60)
                print("  PHASE 3: Motor 2_1 + 2_2 (X axis, stepper_x + stepper_x1)")
                print("  stepper_x  (primary):  step=PF11  dir=PG3   enable=PG5")
                print("  stepper_x1 (secondary): step=PG4   dir=!PC1  enable=PA0")
                print("  Both on same Klipper rail -- always move together")
                print("=" * 60)
                print()

                _drain_socket(sock)
                spin_x_back_and_forth(sock, args.cycles, args.speed)

                _drain_socket(sock)
                print()
                print("  Motor 2_1 + 2_2 test complete.")
                print()

            # --- Phase 4: Endstop verification ----------------------------
            endstops_ok = run_endstop_test(sock)

            if not endstops_ok:
                print("  Endstop verification failed -- skipping homing & motion.")
                print("  Fix endstops and re-run.")
                return
        else:
            print("  [SKIP] Phases 1-4 (--skip-to-homing)")
            print()

        # --- Phase 5: Homing (G28 X Y) -----------------------------------
        homed = home_all(sock)

        if not homed:
            print("  Homing failed -- skipping circle test.")
        else:
            # --- Phase 6: Circle + speed ramp test ------------------------
            run_motion_test(sock)

            # Disable motors after motion test
            _raw_gcode(sock, "M18")  # disable all steppers

    except KeyboardInterrupt:
        print("\n\n  Interrupted -- disabling all motors...")
        try:
            _raw_gcode(sock, "M18")  # disables all kinematic + extra steppers
        except Exception:
            pass

    finally:
        sock.close()

    # --- Summary ----------------------------------------------------------
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print()
    print("  Checklist:")
    print("    [ ] Motor 0 (Y): spun smoothly in both directions?")
    if not args.skip_motor1:
        print("    [ ] Motor 1:     spun smoothly? (skip if not connected)")
    print("    [ ] Motor 2_1+2_2 (X): both spun together, same direction?")
    print("    [ ] Y endstop: open when released, TRIGGERED when pressed?")
    print("    [ ] X endstop: open when released, TRIGGERED when pressed?")
    print("    [ ] G28 X Y: both X motors homed and stopped at endstop?")
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
        # Emergency: disable all motors
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(SOCKET_PATH)
            _raw_gcode(sock, "M18")
            sock.close()
        except OSError:
            pass
        sys.exit(1)
