#!/usr/bin/env python3
"""Hardware bring-up: motor spin, endstop verification, homing, and circle test.

Writes a ``printer.cfg`` using ``kinematics: cartesian`` so Klipper manages
coordinated motion natively.  The dual-X axis uses ``[stepper_x]`` +
``[stepper_x1]`` -- Klipper's native multi-stepper rail mechanism.
Both motors are on the same kinematic rail, so they move in perfect
hardware-level sync for ALL operations (G1, G28, FORCE_MOVE).

The Z axis drives a seesaw belt with effectors on both sides (they move
in opposite directions).  Total mechanical travel is 80 mm with a 1 mm
software buffer on each side (effective 1..79 mm).  A physical limit
switch on DIAG2 (PG10) homes Z to position 0 at reduced speed.

Phases
------
    Phase 1 -- Motor 0 (Y = stepper_y):       FORCE_MOVE spin test
    Phase 2 -- Motor 1 (Z seesaw = stepper_z): FORCE_MOVE spin test
    Phase 3 -- Motor 2_1 + 2_2 (X dual):      G1 spin test (synced)
    Phase 4 -- Endstop verification:           QUERY_ENDSTOPS polling (X, Y, Z)
    Phase 5 -- Homing:                         G28 X Y then G28 Z (slower)
    Phase 6 -- Circle + Z oscillation test:    G2 arcs in XY with helical Z

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

# Ensure the project root is on sys.path so `robot_control` and `src`
# are importable when this script is invoked directly
# (e.g. `python robot_control/scripts/test_motors.py`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg

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
    "motor_z": {  # Motor 1 -- DRIVER_1 slot -- Z seesaw belt
        "step": "PG0", "dir": "PG1", "enable": "PF15",
        "endstop_pin": "PG10",  # DIAG2 / STOP_2 -- Z limit switch
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
WORKSPACE_Z_MM = 80.0   # seesaw belt total mechanical travel

# Z software buffer: 5 mm from each physical limit to protect end-of-travel.
# Klipper hard limits are 0..80; these tighter limits are enforced in code.
Z_BUFFER_MM = 5.0
Z_MIN_SAFE = Z_BUFFER_MM                     # 5 mm
Z_MAX_SAFE = WORKSPACE_Z_MM - Z_BUFFER_MM    # 75 mm
Z_CENTER = WORKSPACE_Z_MM / 2.0              # 40 mm

# Homing speed: 0.5 RPS = 16 mm/s with 32 mm rotation_distance
HOMING_SPEED = 16.0

# Z homing speed: slower than XY because the limit switch has less play.
Z_HOMING_SPEED = 10.0


# ---------------------------------------------------------------------------
# Printer.cfg generation -- delegates to robot_control.configs.printer_cfg
# ---------------------------------------------------------------------------


def generate_test_config() -> str:
    """Generate a printer.cfg from machine.yaml via the configs module.

    Loads the default machine.yaml shipped with robot_control, then
    delegates to ``generate_printer_cfg`` for the actual content.
    """
    cfg = load_config()
    return generate_printer_cfg(cfg)


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
    """Connect to Klipper UDS and wait until state is 'ready'.

    If Klipper reports an error or shutdown state (e.g. "Lost
    communication with MCU"), a ``FIRMWARE_RESTART`` is attempted
    once before continuing to poll.
    """
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
    distance: float | None = None,
) -> None:
    """Spin a cartesian stepper back and forth using FORCE_MOVE.

    FORCE_MOVE moves a stepper by a relative distance without requiring
    homing.  Each cycle: forward *distance*, pause, reverse.

    Parameters
    ----------
    stepper_name : str
        Klipper stepper name, e.g. ``stepper_y``, ``stepper_z``.
    distance : float | None
        Linear travel per direction in mm.  ``None`` defaults to one
        full revolution (``ROTATION_DISTANCE`` = 32 mm).
    """
    dist = distance if distance is not None else ROTATION_DISTANCE
    degrees = (dist / ROTATION_DISTANCE) * 360.0

    for i in range(1, cycles + 1):
        print(f"    Cycle {i}/{cycles}: forward {degrees:.0f}deg "
              f"({dist:.0f} mm) ...", end="", flush=True)
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE={dist} VELOCITY={speed} ACCEL=4000",
        )
        print(" done")

        _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")

        print(f"    Cycle {i}/{cycles}: reverse {degrees:.0f}deg "
              f"({dist:.0f} mm) ...", end="", flush=True)
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE=-{dist} VELOCITY={speed} ACCEL=4000",
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
        # Set only X and Y to 0; omitting Z keeps its current value
        # (Klipper defaults omitted axes to curpos, not 0).
        _raw_gcode(sock, "SET_KINEMATIC_POSITION X=0 Y=0")

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
    z_key = _find_key(initial, "z")

    if not x_key:
        print(f"  !! No X endstop key found in: {list(initial.keys())}")
        return False
    if not y_key:
        print(f"  !! No Y endstop key found in: {list(initial.keys())}")
        return False
    if not z_key:
        print(f"  !! No Z endstop key found in: {list(initial.keys())}")
        return False

    endstops_to_test = [
        (x_key, "X axis endstop (STOP_0 / PG6)"),
        (y_key, "Y axis endstop (STOP_1 / PG9)"),
        (z_key, "Z axis endstop (DIAG2 / PG10)"),
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
    """Home X, Y, and Z using G28.

    stepper_x and stepper_x1 are on the same Klipper rail, so G28
    moves both and both stop when the endstop triggers.

    Z homes separately at reduced speed (5 mm/s vs 16 mm/s for XY)
    because the seesaw limit switch has less mechanical play.

    After homing:
      X = 0   (endstop at min,  workspace 0..200)
      Y = 200 (endstop at max,  workspace 0..200)
      Z = 80  (endstop at max,  workspace 0..80)
    """
    print()
    print("=" * 60)
    print("  PHASE 5: HOMING  (G28 X Y, then G28 Z at reduced speed)")
    print("=" * 60)
    print()

    # Home X and Y first (faster, 16 mm/s)
    print("  Homing X (both motors) and Y ...")
    ok_xy = _raw_gcode(sock, "G28 X Y", timeout=60.0)
    if ok_xy:
        print(f"    Homed OK  X=0  Y={WORKSPACE_Y_MM}")
    else:
        print("    XY homing FAILED")
        print()
        return False

    # Home Z separately (slower due to delicate limit switch)
    print(f"  Homing Z (seesaw, {Z_HOMING_SPEED} mm/s) ...")
    ok_z = _raw_gcode(sock, "G28 Z", timeout=60.0)
    if not ok_z:
        print("    Z homing FAILED")
        print()
        return False

    # Retract Z from the hard stop (position 80) to safe limit.
    # After homing to max, the carriage sits right at the physical end
    # of travel.  Back off immediately to avoid stressing the belt.
    print(f"    Homed OK  Z={WORKSPACE_Z_MM:.0f}, "
          f"retracting to {Z_MAX_SAFE:.0f} ...")
    move_z(sock, Z_MAX_SAFE, feedrate=300.0)
    _raw_gcode(sock, "M400")  # wait for retract to physically complete
    print(f"    Workspace: 0..{WORKSPACE_X_MM:.0f} mm (X)  x  "
          f"0..{WORKSPACE_Y_MM:.0f} mm (Y)  x  "
          f"0..{WORKSPACE_Z_MM:.0f} mm (Z)")
    print(f"    Z safe range: {Z_MIN_SAFE:.0f}..{Z_MAX_SAFE:.0f} mm "
          f"({Z_BUFFER_MM:.0f} mm buffer each side)")

    print()
    return True


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


def move_z(
    sock: socket.socket,
    z: float,
    feedrate: float = 600.0,
) -> None:
    """Move Z axis to *z*, clamped to the safe buffer range [1..79 mm].

    Parameters
    ----------
    feedrate : float
        Speed in mm/min (G-code F parameter).  600 = 10 mm/s.
    """
    z = max(Z_MIN_SAFE, min(z, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.2f} F{feedrate:.0f}")


def move_xyz(
    sock: socket.socket,
    x: float,
    y: float,
    z: float,
    feedrate: float = 1200.0,
) -> None:
    """Move all three axes simultaneously with a single G1 command.

    X/Y clamped to workspace, Z clamped to safe buffer range.
    """
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    z = max(Z_MIN_SAFE, min(z, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate:.0f}")


def draw_circle(
    sock: socket.socket,
    center_x: float,
    center_y: float,
    radius: float,
    feedrate: float = 960.0,
    z_center: float | None = None,
    z_amplitude: float = 0.0,
) -> None:
    """Draw a circle using G2 arc commands, optionally with Z oscillation.

    Uses two G2 semicircles (universally supported by G-code
    interpreters).  Klipper's ``[gcode_arcs]`` module internally
    interpolates each arc into smooth micro-segments with seamless
    velocity blending -- no deceleration between segments, so the
    motion is silent and vibration-free at any speed.

    When *z_center* and *z_amplitude* are given, each semicircle
    includes a linear Z ramp (helical arc), producing a triangle-wave
    Z oscillation of +/- *z_amplitude* around *z_center*.

    Parameters
    ----------
    center_x, center_y : float
        Circle centre in mm (XY plane).
    radius : float
        Circle radius in mm (XY plane).
    feedrate : float
        Travel speed in mm/min (960 = 16 mm/s).
    z_center : float | None
        Centre of Z oscillation in mm.  ``None`` disables Z motion.
    z_amplitude : float
        Half-amplitude of Z oscillation in mm.  Clamped to stay
        within [Z_MIN_SAFE, Z_MAX_SAFE].
    """
    speed_mmps = feedrate / 60.0
    use_z = z_center is not None and z_amplitude > 0.0

    # Clamp Z endpoints to the hard software buffer [1..79 mm].
    z_lo = max(z_center - z_amplitude, Z_MIN_SAFE) if use_z else 0.0
    z_hi = min(z_center + z_amplitude, Z_MAX_SAFE) if use_z else 0.0

    z_tag = f"  Z: {z_lo:.0f} to {z_hi:.0f} mm" if use_z else ""
    print(f"  Drawing circle: centre=({center_x},{center_y}) "
          f"r={radius} mm  {speed_mmps:.0f} mm/s  (G2 arcs){z_tag}")

    # Move ALL axes to start position in a single G1 so they travel
    # simultaneously (3 o'clock position, Z at low end of oscillation).
    start_x = center_x + radius
    start_y = center_y
    if use_z:
        move_xyz(sock, start_x, start_y, z_lo, feedrate)
    else:
        move_xy(sock, start_x, start_y, feedrate)

    # Two semicircles: 3 o'clock -> 9 o'clock -> 3 o'clock
    # I, J = offset from current position to arc centre.
    # Z ramps linearly inside each arc (helical motion), so all three
    # axes move simultaneously during the entire circle.
    #
    # First half: Z ramps z_lo -> z_hi
    z_part_1 = f" Z{z_hi:.2f}" if use_z else ""
    _raw_gcode(
        sock,
        f"G2 X{center_x - radius:.2f} Y{center_y:.2f}"
        f"{z_part_1} "
        f"I{-radius:.2f} J0 F{feedrate:.0f}",
    )

    # Second half: Z ramps z_hi -> z_lo
    z_part_2 = f" Z{z_lo:.2f}" if use_z else ""
    _raw_gcode(
        sock,
        f"G2 X{center_x + radius:.2f} Y{center_y:.2f}"
        f"{z_part_2} "
        f"I{radius:.2f} J0 F{feedrate:.0f}",
    )

    print(f"    circle done")


def run_motion_test(sock: socket.socket) -> None:
    """Phase 6: helical circles (XY + Z) with progressive speed ramp.

    Every circle is a helical arc: XY traces a 40 mm-radius circle
    while Z does a full back-and-forth across its safe range (1..79 mm)
    via linear Z interpolation inside the G2 arcs.  All three axes
    move simultaneously in every command -- no sequential moves.

    Z naturally moves slower than XY because its travel distance per
    semicircle (78 mm) is shorter than XY's arc length (~126 mm).
    At any given feedrate, Z speed is ~62 % of XY tangential speed.
    """
    print()
    print("=" * 60)
    print("  PHASE 6: HELICAL CIRCLE TEST (XY + Z)")
    print("=" * 60)
    print()

    cx = WORKSPACE_X_MM / 2.0   # 100 mm
    cy = WORKSPACE_Y_MM / 2.0   # 100 mm
    radius = 40.0               # 40 mm radius, well within XY bounds

    # Z oscillates across the full safe range (1..79 mm).
    # Per semicircle Z travels 78 mm while XY covers ~126 mm,
    # so Z speed is proportionally slower -- exactly what we want.
    z_amp = Z_CENTER - Z_MIN_SAFE  # 35 mm -> range 5..75 mm

    # Move all three axes to workspace centre simultaneously
    print(f"  Moving to workspace centre "
          f"({cx:.0f}, {cy:.0f}, Z={Z_CENTER:.0f}) ...")
    move_xyz(sock, cx, cy, Z_CENTER, feedrate=1800.0)
    _raw_gcode(sock, "M400")  # wait for physical move to complete
    print("    Arrived.")
    print()

    print(f"  Z back-and-forth: {Z_MIN_SAFE:.0f}..{Z_MAX_SAFE:.0f} mm "
          f"({Z_MAX_SAFE - Z_MIN_SAFE:.0f} mm per half-circle)")
    print()

    # --- Slow verification circle (XYZ) ---
    print("  --- Slow helical circle (16 mm/s) ---")
    draw_circle(
        sock, cx, cy, radius, feedrate=960.0,
        z_center=Z_CENTER, z_amplitude=z_amp,
    )
    print()

    # --- Speed ramp (XYZ) ---
    ramp_speeds = [50, 100, 160, 220, 250, 280, 310]  # mm/s
    print(f"  --- Speed ramp: {ramp_speeds} mm/s  (G2 arcs + Z) ---")
    print(f"  (Ctrl-C to abort if motion becomes rough)")
    print()

    for speed in ramp_speeds:
        feedrate = speed * 60.0
        print(f"  --- {speed} mm/s ---")
        draw_circle(
            sock, cx, cy, radius, feedrate=feedrate,
            z_center=Z_CENTER, z_amplitude=z_amp,
        )
        print()

    # Return all axes to centre simultaneously
    print("  Returning to centre ...")
    move_xyz(sock, cx, cy, Z_CENTER, feedrate=1800.0)
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
    args = parser.parse_args()

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
    print(f"    endstop polarity:   from machine.yaml")
    print()

    # --- Write test config and restart Klipper ----------------------------
    if not args.no_config_write:
        # Back up existing config
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            print(f"  Backed up existing printer.cfg -> {backup}")

        config_text = generate_test_config()
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

                # --- Phase 2: Motor 1 (Z seesaw = stepper_z) --------------
                z_quarter_turn = ROTATION_DISTANCE / 4.0  # 8 mm
                if not args.skip_motor1:
                    print()
                    print("=" * 60)
                    print("  PHASE 2: Motor 1 (DRIVER_1 = stepper_z seesaw)")
                    print("  Pins: step=PG0  dir=PG1  enable=PF15")
                    print("  Endstop: PG10 (DIAG2)")
                    print(f"  Total travel: {WORKSPACE_Z_MM:.0f} mm, "
                          f"safe range: {Z_MIN_SAFE:.0f} to {Z_MAX_SAFE:.0f} mm")
                    print(f"  Using quarter-turn ({z_quarter_turn:.0f} mm) "
                          f"due to limited travel")
                    print("=" * 60)
                    print()
                    print("  WARNING: Centre the Z seesaw manually before running!")
                    print(f"  FORCE_MOVE will travel +/-{z_quarter_turn} mm "
                          f"(90deg).")
                    print()

                    spin_single_stepper(
                        sock, "stepper_z", args.cycles, args.speed,
                        distance=z_quarter_turn,
                    )
                    print()
                    print("  Motor 1 (Z seesaw) test complete.")
                    print("  Verify: both effectors moved in opposite directions.")
                    print()
                else:
                    print()
                    print("  [SKIP] Motor 1 / Z seesaw (--skip-motor1)")
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
            # Fire-and-forget: send M18 without waiting for Klipper's
            # response so Ctrl-C exits immediately instead of blocking
            # up to 30 s on _raw_send's timeout loop.
            payload = (
                json.dumps({
                    "id": 9999,
                    "method": "gcode/script",
                    "params": {"script": "M18"},
                }).encode() + ETX
            )
            sock.settimeout(1.0)
            sock.sendall(payload)
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
        print("    [ ] Motor 1 (Z): spun smoothly? Both effectors moved "
              "opposite?")
    print("    [ ] Motor 2_1+2_2 (X): both spun together, same direction?")
    print("    [ ] X endstop: open when released, TRIGGERED when pressed?")
    print("    [ ] Y endstop: open when released, TRIGGERED when pressed?")
    print("    [ ] Z endstop: open when released, TRIGGERED when pressed?")
    print("    [ ] G28 X Y: both X motors homed and stopped at endstop?")
    print("    [ ] G28 Z: Z homed at reduced speed, stopped at endstop?")
    print("    [ ] Circles: XY smooth at all speeds?")
    print("    [ ] Helical: Z oscillated smoothly during circles?")
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
        # Emergency: fire-and-forget M18 with short timeout
        try:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.settimeout(2.0)
            _sock.connect(SOCKET_PATH)
            _payload = (
                json.dumps({
                    "id": 9998,
                    "method": "gcode/script",
                    "params": {"script": "M18"},
                }).encode() + ETX
            )
            _sock.sendall(_payload)
            _sock.close()
        except OSError:
            pass
        sys.exit(1)
