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

Motor parameters (read from machine.yaml at startup)
-----------------------------------------------------
    All motor, workspace, and speed parameters come from machine.yaml.
    Nothing is hardcoded -- edit the YAML to change behaviour.

Usage::

    .venv/bin/python robot_control/scripts/test_motors.py
    .venv/bin/python robot_control/scripts/test_motors.py --skip-motor1
    .venv/bin/python robot_control/scripts/test_motors.py --endstops-only
    .venv/bin/python robot_control/scripts/test_motors.py --speed 60
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import select
import socket
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `robot_control` and `src`
# are importable when this script is invoked directly
# (e.g. `.venv/bin/python robot_control/scripts/test_motors.py`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_control.configs.loader import load_config
from robot_control.configs.printer_cfg import generate_printer_cfg
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- ALL derived from machine.yaml via load_config()
# ---------------------------------------------------------------------------

# Load once at import time; every constant below comes from here.
_CFG = load_config()

SOCKET_PATH = "/tmp/klippy_uds"
PRINTER_CFG_PATH = Path.home() / "printer.cfg"
ETX = b"\x03"
_SOCKET_RECV_BUFFER_SIZE = 4096

# Motor parameters (from steppers section)
ROTATION_DISTANCE = _CFG.steppers.xy_rotation_distance
MICROSTEPS = _CFG.steppers.klipper_microsteps
DIRECTION_PAUSE_MS = int(_CFG.steppers.direction_reversal_pause_s * 1000)

# step_pulse_duration MUST be written as a fixed-point string to avoid
# Python's scientific notation (5e-06) in the generated printer.cfg.
STEP_PULSE_DURATION = f"{_CFG.steppers.step_pulse_duration_s:.6f}"

# Pin assignments -- read from machine.yaml axes section.
# Endstop pins: bare pin name; prefix (^ or ^!) added in generate_test_config.
MOTOR_PINS = {
    "motor_y": {
        "step": _CFG.axes["y"].pins[0].step,
        "dir": _CFG.axes["y"].pins[0].dir,
        "enable": _CFG.axes["y"].pins[0].enable,
        "endstop_pin": _CFG.axes["y"].endstop_pin,
    },
    "motor_z": {
        "step": _CFG.axes["z"].pins[0].step,
        "dir": _CFG.axes["z"].pins[0].dir,
        "enable": _CFG.axes["z"].pins[0].enable,
        "endstop_pin": _CFG.axes["z"].endstop_pin,
    },
    "motor_x1": {
        "step": _CFG.axes["x"].pins[0].step,
        "dir": _CFG.axes["x"].pins[0].dir,
        "enable": _CFG.axes["x"].pins[0].enable,
        "endstop_pin": _CFG.axes["x"].endstop_pin,
    },
    "motor_x2": {
        "step": _CFG.axes["x"].pins[1].step,
        "dir": _CFG.axes["x"].pins[1].dir,
        "enable": _CFG.axes["x"].pins[1].enable,
        "endstop_pin": None,
    },
}

# The two X motors (motor_x1, motor_x2) are on the same gantry bar.
# They MUST always be enabled, moved, and disabled together.
X_MOTORS = ("motor_x1", "motor_x2")

# Workspace bounds (mm) -- from machine.yaml work_area_mm.
WORKSPACE_X_MM = _CFG.work_area.x
WORKSPACE_Y_MM = _CFG.work_area.y
WORKSPACE_Z_MM = _CFG.work_area.z

# Z software buffer: 5 mm from each physical limit to protect end-of-travel.
# Klipper hard limits are 0..WORKSPACE_Z_MM; tighter limits enforced in code.
Z_BUFFER_MM = 5.0
Z_MIN_SAFE = Z_BUFFER_MM
Z_MAX_SAFE = WORKSPACE_Z_MM - Z_BUFFER_MM
Z_CENTER = WORKSPACE_Z_MM / 2.0

# Homing speeds (from motion section)
HOMING_SPEED = _CFG.motion.homing_speed_mm_s
Z_HOMING_SPEED = _CFG.motion.z_homing_speed_mm_s


# ---------------------------------------------------------------------------
# Printer.cfg generation -- delegates to robot_control.configs.printer_cfg
# ---------------------------------------------------------------------------


def generate_test_config() -> str:
    """Generate a printer.cfg from machine.yaml via the configs module.

    Reuses the module-level ``_CFG`` loaded at import time, so the
    generated printer.cfg is always consistent with the constants
    used by the test functions.
    """
    return generate_printer_cfg(_CFG)


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
                chunk = sock.recv(_SOCKET_RECV_BUFFER_SIZE)
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
            data = sock.recv(_SOCKET_RECV_BUFFER_SIZE)
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
        logger.error("G-code error: %s", msg[:120])
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
                logger.warning("Klipper state: %s -- %s", state, state_msg[:80])
                logger.info("Attempting FIRMWARE_RESTART...")
                try:
                    _raw_gcode(sock, "FIRMWARE_RESTART")
                except (OSError, socket.timeout):
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
        logger.info(
            "Cycle %d/%d: forward %.0f deg (%.0f mm) ...",
            i, cycles, degrees, dist,
        )
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE={dist} VELOCITY={speed} ACCEL=4000",
        )
        logger.info("done")

        _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")

        logger.info(
            "Cycle %d/%d: reverse %.0f deg (%.0f mm) ...",
            i, cycles, degrees, dist,
        )
        _raw_gcode(
            sock,
            f"FORCE_MOVE STEPPER={stepper_name} "
            f"DISTANCE=-{dist} VELOCITY={speed} ACCEL=4000",
        )
        logger.info("done")

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
        logger.info("Cycle %d/%d: forward 360deg (both) ...", i, cycles)
        _raw_gcode(sock, f"G1 X{one_rev:.1f} F{feedrate:.0f}")
        logger.info("done")

        _raw_gcode(sock, f"G4 P{DIRECTION_PAUSE_MS}")
        _drain_socket(sock)

        logger.info("Cycle %d/%d: reverse 360deg (both) ...", i, cycles)
        _raw_gcode(sock, f"G1 X0 F{feedrate:.0f}")
        logger.info("done")

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
        logger.debug("objects/query raw response: %s", json.dumps(resp, indent=2))

    results: dict[str, str] = {}
    status = resp.get("result", {}).get("status", {})
    last_query = status.get("query_endstops", {}).get("last_query", {})

    if debug and not last_query:
        logger.debug(
            "last_query is empty -- objects/query may have returned "
            "a stale notification instead of the actual response.",
        )

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
    except (OSError, ValueError):
        return False


def diagnose_mcu(sock: socket.socket) -> None:
    """Query and print MCU build info to verify firmware matches hardware.

    Queries the Klipper object model for MCU version, build config,
    and pin capabilities.  This helps diagnose whether the firmware
    was compiled for the correct MCU (STM32H723) and whether GPIO
    pins are accessible.
    """
    logger.info("--- MCU Firmware Diagnostic ---")

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

        logger.info("MCU type:      %s", mcu_name)
        if freq:
            logger.info("Clock freq:    %s Hz", f"{freq:,}")
        else:
            logger.info("Clock freq:    unknown")
        logger.info("FW version:    %s", version)
        logger.info("Build:         %s", str(build)[:80])

        # Check if it's actually H723
        if "stm32h723" in str(mcu_name).lower():
            logger.info("Firmware is built for STM32H723")
        elif "stm32" in str(mcu_name).lower():
            logger.warning(
                "Firmware is for %s, NOT H723! "
                "This could cause wrong pin mappings.", mcu_name,
            )
        else:
            logger.info("MCU type:      %s", mcu_name)

    except (OSError, socket.timeout, json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to query MCU info: %s", exc)

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

        logger.info("Loaded endstop pins from config:")
        for section_name, section_data in config.items():
            if section_name.startswith(("stepper_", "manual_stepper")):
                ep = section_data.get("endstop_pin", None)
                if ep:
                    logger.info("  [%s] endstop_pin = %s", section_name, ep)

    except (OSError, socket.timeout, json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to query config: %s", exc)



def run_endstop_test(sock: socket.socket) -> bool:
    """Quick endstop verification via live polling.

    For each endstop, polls QUERY_ENDSTOPS every 0.3s.  Auto-advances
    to the next endstop as soon as an OPEN -> TRIGGERED transition is
    detected.  Press Enter to skip if an endstop is stuck.

    Returns True if all endstops passed.
    """
    logger.info("=" * 60)
    logger.info("  PHASE 4: ENDSTOP VERIFICATION")
    logger.info("=" * 60)
    logger.info("  Press each endstop when prompted.  The test auto-advances")
    logger.info("  once it detects the OPEN -> TRIGGERED transition.")
    logger.info("  Press Enter to skip a stuck endstop.")

    # Discover endstop key names from Klipper (varies by kinematics).
    # cartesian: "x", "y", "z"  or  "stepper_x", "stepper_y", "stepper_z"
    # manual_stepper: "manual_stepper motor_y", etc.
    try:
        initial = query_endstops(sock)
    except (OSError, socket.timeout, json.JSONDecodeError) as exc:
        logger.error("Could not query endstops: %s", exc)
        return False

    logger.info("Detected endstop keys: %s", list(initial.keys()))

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
        logger.error("No X endstop key found in: %s", list(initial.keys()))
        return False
    if not y_key:
        logger.error("No Y endstop key found in: %s", list(initial.keys()))
        return False
    if not z_key:
        logger.error("No Z endstop key found in: %s", list(initial.keys()))
        return False

    endstops_to_test = [
        (x_key, "X axis endstop (STOP_0 / PG6)"),
        (y_key, "Y axis endstop (STOP_1 / PG9)"),
        (z_key, "Z axis endstop (DIAG2 / PG10)"),
    ]

    all_pass = True

    for endstop_name, description in endstops_to_test:
        logger.info(">> %s  [key: %s]", description, endstop_name)
        logger.info("Release the switch, then press it.")

        saw_open = False
        saw_triggered = False

        while True:
            try:
                states = query_endstops(sock)
            except (OSError, socket.timeout, json.JSONDecodeError) as exc:
                logger.error("Query error: %s", exc)
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
                logger.info("PASS  %s", endstop_name)
                break

            # Allow Enter to skip
            if _stdin_has_data():
                sys.stdin.readline()
                if saw_open and not saw_triggered:
                    logger.warning("FAIL  %s: saw OPEN but never TRIGGERED", endstop_name)
                elif saw_triggered and not saw_open:
                    logger.warning("FAIL  %s: always TRIGGERED, never OPEN", endstop_name)
                else:
                    logger.info("SKIP  %s", endstop_name)
                all_pass = False
                break

            time.sleep(0.3)


    tag = "ALL PASSED" if all_pass else "SOME FAILED"
    logger.info("ENDSTOP RESULT: %s", tag)
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
    logger.info("=" * 60)
    logger.info("  PHASE 5: HOMING  (G28 X Y, then G28 Z at reduced speed)")
    logger.info("=" * 60)

    # Home X and Y first (faster, 16 mm/s)
    logger.info("  Homing X (both motors) and Y ...")
    ok_xy = _raw_gcode(sock, "G28 X Y", timeout=60.0)
    if ok_xy:
        logger.info("Homed OK  X=0  Y=%s", WORKSPACE_Y_MM)
    else:
        logger.info("    XY homing FAILED")
        return False

    # Home Z separately (slower due to delicate limit switch)
    logger.info("Homing Z (seesaw, %s mm/s) ...", Z_HOMING_SPEED)
    ok_z = _raw_gcode(sock, "G28 Z", timeout=60.0)
    if not ok_z:
        logger.info("    Z homing FAILED")
        return False

    # Retract Z from the hard stop (position 80) to safe limit.
    # After homing to max, the carriage sits right at the physical end
    # of travel.  Back off immediately to avoid stressing the belt.
    logger.info("Homed OK  Z=%.0f, retracting to %.0f ...", WORKSPACE_Z_MM, Z_MAX_SAFE)
    move_z(sock, Z_MAX_SAFE, feedrate=600.0)
    _raw_gcode(sock, "M400")  # wait for retract to physically complete
    logger.info(
        "Workspace: 0..%.0f mm (X)  x  0..%.0f mm (Y)  x  0..%.0f mm (Z)",
        WORKSPACE_X_MM, WORKSPACE_Y_MM, WORKSPACE_Z_MM,
    )
    logger.info(
        "Z safe range: %.0f..%.0f mm (%.0f mm buffer each side)",
        Z_MIN_SAFE, Z_MAX_SAFE, Z_BUFFER_MM,
    )

    return True


def move_xy(
    sock: socket.socket,
    x: float,
    y: float,
    feedrate: float = 2400.0,
) -> None:
    """Move to (x, y) using G1.  Both X motors move in sync.

    Parameters
    ----------
    feedrate : float
        Speed in mm/min (G-code F parameter).  2400 = 40 mm/s.
    """
    x = max(0.0, min(x, WORKSPACE_X_MM))
    y = max(0.0, min(y, WORKSPACE_Y_MM))
    _raw_gcode(sock, f"G1 X{x:.2f} Y{y:.2f} F{feedrate:.0f}")


def move_z(
    sock: socket.socket,
    z: float,
    feedrate: float = 1200.0,
) -> None:
    """Move Z axis to *z*, clamped to the safe buffer range [5..75 mm].

    Parameters
    ----------
    feedrate : float
        Speed in mm/min (G-code F parameter).  1200 = 20 mm/s.
    """
    z = max(Z_MIN_SAFE, min(z, Z_MAX_SAFE))
    _raw_gcode(sock, f"G1 Z{z:.2f} F{feedrate:.0f}")


def move_xyz(
    sock: socket.socket,
    x: float,
    y: float,
    z: float,
    feedrate: float = 2400.0,
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
    feedrate: float = 1920.0,
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
    logger.info(
        "Drawing circle: centre=(%s,%s) r=%s mm  %.0f mm/s  (G2 arcs)%s",
        center_x, center_y, radius, speed_mmps, z_tag,
    )

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

    logger.info("circle done")


def run_limit_reach_test(sock: socket.socket) -> None:
    """Phase 5b: move each axis to within 10 mm of both limits.

    Verifies that the full workspace is reachable after homing.
    The operator can press Enter at any time to skip the remainder.

    Order: X min -> X max -> Y min -> Y max -> Z min -> Z max -> centre.
    """
    MARGIN = 10.0  # mm from each workspace edge

    x_lo = MARGIN
    x_hi = WORKSPACE_X_MM - MARGIN
    y_lo = MARGIN
    y_hi = WORKSPACE_Y_MM - MARGIN
    z_lo = max(Z_MIN_SAFE, MARGIN)
    z_hi = min(Z_MAX_SAFE, WORKSPACE_Z_MM - MARGIN)

    cx = WORKSPACE_X_MM / 2.0
    cy = WORKSPACE_Y_MM / 2.0
    feedrate = 12000.0  # 200 mm/s

    logger.info("=" * 60)
    logger.info("  PHASE 5b: LIMIT REACH TEST (each axis, both sides)")
    logger.info("=" * 60)
    logger.info("Margin: %.0f mm from each limit", MARGIN)
    logger.info("  X: %.0f .. %.0f mm", x_lo, x_hi)
    logger.info("  Y: %.0f .. %.0f mm", y_lo, y_hi)
    logger.info("  Z: %.0f .. %.0f mm", z_lo, z_hi)
    logger.info("  Press Enter at any time to skip remaining moves.")

    skipped = False

    def _check_skip() -> bool:
        """Return True if the operator pressed Enter to skip."""
        if _stdin_has_data():
            sys.stdin.readline()
            return True
        return False

    # Start from workspace centre
    logger.info("  Moving to centre ...")
    move_xyz(sock, cx, cy, Z_CENTER, feedrate)
    _raw_gcode(sock, "M400")

    # Each move block: issue move, wait, check for skip
    moves = [
        (x_lo, cy,  Z_CENTER, f"X -> {x_lo:.0f} mm (min + {MARGIN:.0f})"),
        (x_hi, cy,  Z_CENTER, f"X -> {x_hi:.0f} mm (max - {MARGIN:.0f})"),
        (cx,   y_lo, Z_CENTER, f"Y -> {y_lo:.0f} mm (min + {MARGIN:.0f})"),
        (cx,   y_hi, Z_CENTER, f"Y -> {y_hi:.0f} mm (max - {MARGIN:.0f})"),
        (cx,   cy,  z_lo,     f"Z -> {z_lo:.0f} mm (min + {MARGIN:.0f})"),
        (cx,   cy,  z_hi,     f"Z -> {z_hi:.0f} mm (max - {MARGIN:.0f})"),
    ]

    for mx, my, mz, label in moves:
        if _check_skip():
            skipped = True
            break
        logger.info("%s ...", label)
        move_xyz(sock, mx, my, mz, feedrate)
        _raw_gcode(sock, "M400")

    if skipped:
        logger.info("  [SKIP] Remaining limit moves skipped by operator.")

    # Always return to centre before next phase
    logger.info("  Returning to centre ...")
    move_xyz(sock, cx, cy, Z_CENTER, feedrate)
    _raw_gcode(sock, "M400")
    logger.info("    Done.")


def run_motion_test(sock: socket.socket) -> None:
    """Phase 6: helical circles (XY + Z) with progressive speed ramp.

    Every circle is a helical arc: XY traces a circle (radius = 30%
    of the shorter XY axis) while Z does a full back-and-forth across
    its safe range via linear Z interpolation inside the G2 arcs.
    All three axes move simultaneously -- no sequential moves.
    """
    logger.info("=" * 60)
    logger.info("  PHASE 6: HELICAL CIRCLE TEST (XY + Z)")
    logger.info("=" * 60)

    cx = WORKSPACE_X_MM / 2.0
    cy = WORKSPACE_Y_MM / 2.0
    # 30% of the shorter XY axis, so the circle fills the workspace visibly
    # while leaving plenty of margin from the limits.
    radius = min(WORKSPACE_X_MM, WORKSPACE_Y_MM) * 0.30

    # Z oscillates across the full safe range.
    # Per semicircle Z travels less distance than XY's arc length,
    # so Z speed is proportionally slower -- exactly what we want.
    z_amp = Z_CENTER - Z_MIN_SAFE  # 35 mm -> range 5..75 mm

    # Move all three axes to workspace centre simultaneously
    logger.info(
        "Moving to workspace centre (%.0f, %.0f, Z=%.0f) ...",
        cx, cy, Z_CENTER,
    )
    move_xyz(sock, cx, cy, Z_CENTER, feedrate=3600.0)
    _raw_gcode(sock, "M400")  # wait for physical move to complete
    logger.info("    Arrived.")

    logger.info(
        "Z back-and-forth: %.0f..%.0f mm (%.0f mm per half-circle)",
        Z_MIN_SAFE, Z_MAX_SAFE, Z_MAX_SAFE - Z_MIN_SAFE,
    )

    # --- Slow verification circle (XYZ) ---
    logger.info("  --- Slow helical circle (32 mm/s) ---")
    draw_circle(
        sock, cx, cy, radius, feedrate=1920.0,
        z_center=Z_CENTER, z_amplitude=z_amp,
    )

    # --- Speed ramp (XYZ) ---
    ramp_speeds = [100, 200, 320, 440, 500, 560, 620]  # mm/s
    logger.info("--- Speed ramp: %s mm/s  (G2 arcs + Z) ---", ramp_speeds)
    logger.info("(Ctrl-C to abort if motion becomes rough)")

    for speed in ramp_speeds:
        feedrate = speed * 60.0
        logger.info("--- %d mm/s ---", speed)
        draw_circle(
            sock, cx, cy, radius, feedrate=feedrate,
            z_center=Z_CENTER, z_amplitude=z_amp,
        )

    # Return all axes to centre simultaneously
    logger.info("  Returning to centre ...")
    move_xyz(sock, cx, cy, Z_CENTER, feedrate=3600.0)
    logger.info("    Done.")


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
    setup_logging()

    rps = args.speed / ROTATION_DISTANCE

    logger.info("=" * 60)
    logger.info("  HARDWARE BRING-UP TEST")
    logger.info("=" * 60)
    logger.info("  Motor parameters:")
    logger.info("  rotation_distance:  %s mm/rev", ROTATION_DISTANCE)
    logger.info("  microsteps:         %s  (must match DM542TE DIP)", MICROSTEPS)
    logger.info("  step_pulse:         %s s (5 us)", STEP_PULSE_DURATION)
    full_steps = _CFG.steppers.full_steps_per_rotation
    logger.info(
        "  pulses/rev:         %d  (%d full-steps x %d)",
        full_steps * MICROSTEPS, full_steps, MICROSTEPS,
    )
    logger.info("  test speed:         %s mm/s (%.2f RPS)", args.speed, rps)
    logger.info("  direction pause:    %d ms", DIRECTION_PAUSE_MS)
    logger.info("  cycles:             %d back-and-forth", args.cycles)
    logger.info("  endstop polarity:   from machine.yaml")
    logger.info("  Workspace (from machine.yaml):")
    logger.info("  X: 0 .. %.0f mm", WORKSPACE_X_MM)
    logger.info("  Y: 0 .. %.0f mm", WORKSPACE_Y_MM)
    logger.info(
        "  Z: 0 .. %.0f mm  (safe: %.0f .. %.0f mm)",
        WORKSPACE_Z_MM, Z_MIN_SAFE, Z_MAX_SAFE,
    )

    # --- Write test config and restart Klipper ----------------------------
    if not args.no_config_write:
        # Back up existing config
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
            logger.info("Backed up existing printer.cfg -> %s", backup)

        config_text = generate_test_config()
        PRINTER_CFG_PATH.write_text(config_text)
        logger.info("Wrote test printer.cfg to %s", PRINTER_CFG_PATH)

        logger.info("  Restarting Klipper to load test config...")
        restart_klipper()

    logger.info("  Waiting for Klipper to become ready...")
    sock = _wait_for_ready(timeout=30.0)
    logger.info("  [OK] Klipper is ready")

    # --- MCU diagnostic: verify firmware matches the board -----------------
    diagnose_mcu(sock)

    try:
        if not args.skip_to_homing:
            # --- Phase 1: Motor 0 (Y) ------------------------------------
            if not args.endstops_only:
                logger.info("=" * 60)
                logger.info("  PHASE 1: Motor 0 (Y axis = stepper_y)")
                logger.info("  Pins: step=PF13  dir=PF12  enable=PF14")
                logger.info("  Uses FORCE_MOVE for individual stepper test")
                logger.info("=" * 60)

                spin_single_stepper(sock, "stepper_y", args.cycles, args.speed)
                logger.info("  Motor 0 test complete.")

                # --- Phase 2: Motor 1 (Z seesaw = stepper_z) --------------
                z_quarter_turn = ROTATION_DISTANCE / 4.0  # 8 mm
                if not args.skip_motor1:
                    logger.info("=" * 60)
                    logger.info("  PHASE 2: Motor 1 (DRIVER_1 = stepper_z seesaw)")
                    logger.info("  Pins: step=PG0  dir=PG1  enable=PF15")
                    logger.info("  Endstop: PG10 (DIAG2)")
                    logger.info(
                        "Total travel: %.0f mm, safe range: %.0f to %.0f mm",
                        WORKSPACE_Z_MM, Z_MIN_SAFE, Z_MAX_SAFE,
                    )
                    logger.info(
                        "Using quarter-turn (%.0f mm) due to limited travel",
                        z_quarter_turn,
                    )
                    logger.info("=" * 60)
                    logger.info("  WARNING: Centre the Z seesaw manually before running!")
                    logger.warning(
                        "Centre the Z seesaw manually before running! "
                        "FORCE_MOVE will travel +/-%.0f mm (90deg).",
                        z_quarter_turn,
                    )

                    spin_single_stepper(
                        sock, "stepper_z", args.cycles, args.speed,
                        distance=z_quarter_turn,
                    )
                    logger.info("  Motor 1 (Z seesaw) test complete.")
                    logger.info("  Verify: both effectors moved in opposite directions.")
                else:
                    logger.info("  [SKIP] Motor 1 / Z seesaw (--skip-motor1)")

                # --- Phase 3: Motor 2_1 + 2_2 (X dual, same rail) --------
                logger.info("=" * 60)
                logger.info("  PHASE 3: Motor 2_1 + 2_2 (X axis, stepper_x + stepper_x1)")
                logger.info("  stepper_x  (primary):  step=PF11  dir=PG3   enable=PG5")
                logger.info("  stepper_x1 (secondary): step=PG4   dir=!PC1  enable=PA0")
                logger.info("  Both on same Klipper rail -- always move together")
                logger.info("=" * 60)

                _drain_socket(sock)
                spin_x_back_and_forth(sock, args.cycles, args.speed)

                _drain_socket(sock)
                logger.info("  Motor 2_1 + 2_2 test complete.")

            # --- Phase 4: Endstop verification ----------------------------
            endstops_ok = run_endstop_test(sock)

            if not endstops_ok:
                logger.info("  Endstop verification failed -- skipping homing & motion.")
                logger.info("  Fix endstops and re-run.")
                return
        else:
            logger.info("  [SKIP] Phases 1-4 (--skip-to-homing)")

        # --- Phase 5: Homing (G28 X Y) -----------------------------------
        homed = home_all(sock)

        if not homed:
            logger.info("  Homing failed -- skipping limit reach & circle test.")
        else:
            # --- Phase 5b: Limit reach test -------------------------------
            run_limit_reach_test(sock)

            # --- Phase 6: Circle + speed ramp test ------------------------
            run_motion_test(sock)

            # Disable motors after motion test
            _raw_gcode(sock, "M18")  # disable all steppers

    except KeyboardInterrupt:
        logger.warning("Interrupted -- disabling all motors...")
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
        except OSError:
            pass

    finally:
        sock.close()

    # --- Summary ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  TEST COMPLETE")
    logger.info("=" * 60)
    logger.info("  Checklist:")
    logger.info("    [ ] Motor 0 (Y): spun smoothly in both directions?")
    if not args.skip_motor1:
        logger.info("    [ ] Motor 1 (Z): spun smoothly? Both effectors moved "
              "opposite?")
    logger.info("    [ ] Motor 2_1+2_2 (X): both spun together, same direction?")
    logger.info("    [ ] X endstop: open when released, TRIGGERED when pressed?")
    logger.info("    [ ] Y endstop: open when released, TRIGGERED when pressed?")
    logger.info("    [ ] Z endstop: open when released, TRIGGERED when pressed?")
    logger.info("    [ ] G28 X Y: both X motors homed and stopped at endstop?")
    logger.info("    [ ] G28 Z: Z homed at reduced speed, stopped at endstop?")
    logger.info("    [ ] Limit reach: all axes reached within 10 mm of both limits?")
    logger.info("    [ ] Circles: XY smooth at all speeds?")
    logger.info("    [ ] Helical: Z oscillated smoothly during circles?")

    if not args.no_config_write:
        backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
        if backup.exists():
            logger.info("To restore your production config:")
            logger.info("  mv %s %s", backup, PRINTER_CFG_PATH)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Fatal error: %s", exc)
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
