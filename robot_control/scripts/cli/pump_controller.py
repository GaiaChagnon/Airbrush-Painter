"""Pump controller mode -- persistent Rich layout with real-time status.

Provides all functionality of ``pump_testbed.py`` with a polished
terminal UI: persistent pump status panels, a scrollable activity log,
questionary menus, and ``plotext`` charts for volume calibration.

The rendering model uses ``render_screen()`` (clear + redraw) between
prompts, and ``Rich Live`` during long operations so the pump-status
panel updates while moves execute.
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotext as plt
import questionary
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robot_control.configs.loader import (
    MachineConfig,
    PumpMotorConfig,
    PumpsConfig,
    SyringeConfig,
)
from robot_control.configs.printer_cfg import generate_pump_test_cfg
from robot_control.hardware.pump_control import (
    PRINTER_CFG_PATH,
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
from robot_control.scripts.cli.widgets import info_panel, pump_diagram
from src.utils.fs import atomic_yaml_dump, load_yaml

if TYPE_CHECKING:
    import socket

    from robot_control.scripts.cli.app import RobotApp

_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "machine.yaml"
_MANUAL_CAL_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs" / "sim" / "manual_calibration_results.yaml"
)


# ======================================================================
# Session state with persistent UI
# ======================================================================


class _PumpSession:
    """Runtime state for a pump controller session.

    Maintains a ring-buffer activity log and provides persistent
    screen rendering via ``render_screen()`` (static, before prompts)
    and ``build_live_renderable()`` (for Rich Live during operations).
    """

    _MAX_LOG_LINES = 80

    def __init__(self, app: RobotApp) -> None:
        self.app = app
        self.cfg: MachineConfig = app.config
        self.pumps_cfg: PumpsConfig = self.cfg.pumps  # type: ignore[assignment]
        self.console: Console = app.console
        self.sock: socket.socket | None = None
        self.homed: dict[str, bool] = {pid: False for pid in self.pumps_cfg.motors}
        self.valve_open_flag: bool = False
        self.needle_retracted: bool = False
        self.pump_positions: dict[str, float] = {
            pid: 0.0 for pid in self.pumps_cfg.motors
        }
        self.current_speeds: dict[str, float] = {
            pid: 0.0 for pid in self.pumps_cfg.motors
        }

        self._log_lines: list[str] = []
        self._summary_panel: Panel | None = None
        self._progress_text: str = ""

    @property
    def pump_ids(self) -> list[str]:
        return list(self.pumps_cfg.motors.keys())

    @property
    def ink_pump_ids(self) -> list[str]:
        return [pid for pid, m in self.pumps_cfg.motors.items() if m.fluid != "purge"]

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

    # ------------------------------------------------------------------
    # Persistent UI helpers
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        """Append a Rich-markup message to the activity log ring buffer."""
        self._log_lines.append(msg)
        if len(self._log_lines) > self._MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self._MAX_LOG_LINES :]

    def set_summary(self, panel: Panel | None) -> None:
        """Set or clear the summary panel (shown above the activity log)."""
        self._summary_panel = panel

    def set_progress(self, text: str) -> None:
        """Set or clear the current-operation progress indicator."""
        self._progress_text = text

    def _build_pump_panel(self) -> Panel:
        """Build the pump-status panel from current session state."""
        states: dict[str, dict[str, Any]] = {}
        for pid in self.pump_ids:
            m = self.motor(pid)
            sy = self.syringe(pid)
            states[pid] = {
                "fluid": m.fluid,
                "position_mm": self.pump_positions.get(pid, 0.0),
                "travel_mm": sy.plunger_travel_mm,
                "homed": self.homed[pid],
                "speed_mm_s": self.current_speeds.get(pid, 0.0),
            }
        return pump_diagram(states, self.valve_open_flag, self.needle_retracted)

    def _build_log_panel(self, max_lines: int = 0) -> Panel:
        """Build the activity-log panel from the ring buffer.

        Parameters
        ----------
        max_lines : int
            If > 0, cap at this many lines (used by ``render_screen``
            for a compact view).  Otherwise, adapts to terminal height.
        """
        if max_lines > 0:
            n_visible = max_lines
        else:
            term_h = self.console.height or 40
            overhead = 12 + len(self.pump_ids)
            if self._summary_panel:
                overhead += 8
            if self._progress_text:
                overhead += 3
            n_visible = max(4, term_h - overhead)
        lines = self._log_lines[-n_visible:]

        if not lines:
            content = Text.from_markup("[dim]No activity yet.[/]")
        else:
            parts: list[Text] = []
            for line in lines:
                try:
                    parts.append(Text.from_markup(line))
                except Exception:
                    parts.append(Text(line))
            content = Text("\n").join(parts)

        return Panel(content, title="[bold]Activity Log[/]", border_style="dim")

    def build_live_renderable(self) -> Group:
        """Compose the full layout for a ``Rich Live`` context."""
        parts: list[Any] = [self._build_pump_panel()]
        if self._progress_text:
            parts.append(
                Panel(
                    Text.from_markup(self._progress_text),
                    border_style="yellow",
                    height=3,
                )
            )
        if self._summary_panel:
            parts.append(self._summary_panel)
        parts.append(self._build_log_panel())
        return Group(*parts)

    def render_screen(self) -> None:
        """Clear terminal and render pump panel + compact activity log.

        Pump status stays at the top, a small log panel follows, and
        the questionary prompt appears right below.
        """
        self.console.clear()
        self.console.print(self._build_pump_panel())
        if self._progress_text:
            self.console.print(
                Panel(
                    Text.from_markup(self._progress_text),
                    border_style="yellow",
                    height=3,
                )
            )
        if self._summary_panel:
            self.console.print(self._summary_panel)
        self.console.print(self._build_log_panel(max_lines=6))

    # ------------------------------------------------------------------
    # Valve & safety (output to log buffer)
    # ------------------------------------------------------------------

    def open_valve(self) -> None:
        name = self.pumps_cfg.refill_valve_output
        if not name:
            return
        sock = self.ensure_connected()
        self.log("  Opening refill valve...")
        set_pin(sock, name, 1)
        time.sleep(self.pumps_cfg.refill_valve_delay_s)
        self.valve_open_flag = True
        self.log("  Refill valve [green]OPEN[/]")
        self.app.session_log.log_action("pump", "valve_open")

    def close_valve(self) -> None:
        name = self.pumps_cfg.refill_valve_output
        if not name:
            return
        sock = self.ensure_connected()
        self.log("  Closing refill valve...")
        set_pin(sock, name, 0)
        time.sleep(self.pumps_cfg.refill_valve_delay_s)
        self.valve_open_flag = False
        self.log("  Refill valve [red]CLOSED[/]")
        self.app.session_log.log_action("pump", "valve_close")

    def check_travel_limit(self, pid: str, target_position: float) -> bool:
        if not self.pumps_cfg.enforce_travel_limits:
            return True
        if not self.homed[pid]:
            return True
        sy = self.syringe(pid)
        if abs(target_position) > sy.plunger_travel_mm:
            self.log(
                f"  [red]LIMIT VIOLATION:[/] {pid} target {target_position:.4f} mm "
                f"exceeds plunger travel {sy.plunger_travel_mm:.4f} mm. Move rejected."
            )
            return False
        return True


# ======================================================================
# Input helpers (questionary wrappers)
# ======================================================================


def _ask_float(app: RobotApp, prompt: str, default: float, key: str = "") -> float:
    hist_key = key or prompt
    result = questionary.text(
        f"{prompt} [{default:g}]:",
        default=str(default) if default else "",
    ).ask()
    if result is None:
        raise KeyboardInterrupt
    val = result.strip()
    if not val:
        return default
    try:
        v = float(val)
        app.add_history(hist_key, val)
        return v
    except ValueError:
        return default


def _ask_int(app: RobotApp, prompt: str, default: int) -> int:
    result = questionary.text(f"{prompt} [{default}]:", default=str(default)).ask()
    if result is None:
        raise KeyboardInterrupt
    val = result.strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _ask_confirm(prompt: str, default: bool = True) -> bool:
    result = questionary.confirm(prompt, default=default).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def _select_pumps(
    s: _PumpSession, prompt: str = "Select pumps", allow_all: bool = True
) -> list[str]:
    choices = [f"{s.pump_label(pid)}" for pid in s.pump_ids]
    if allow_all:
        choices.insert(0, "All pumps")
    selected = questionary.checkbox(prompt, choices=choices).ask()
    if selected is None:
        raise KeyboardInterrupt
    if "All pumps" in selected or not selected:
        return list(s.pump_ids)
    return [pid for pid in s.pump_ids if s.pump_label(pid) in selected]


def _select_single_pump(s: _PumpSession, prompt: str = "Select pump") -> str:
    choices = [s.pump_label(pid) for pid in s.pump_ids]
    selected = questionary.select(prompt, choices=choices).ask()
    if selected is None:
        raise KeyboardInterrupt
    for pid in s.pump_ids:
        if s.pump_label(pid) == selected:
            return pid
    return s.pump_ids[0]


def _pause_ms(s: _PumpSession) -> int:
    return int(s.pumps_cfg.stepper.direction_reversal_pause_s * 1000)


def _dwell(s: _PumpSession) -> None:
    raw_gcode(s.ensure_connected(), f"G4 P{_pause_ms(s)}")


def _dispense_sign(m: PumpMotorConfig) -> int:
    return -m.homing_direction


# ======================================================================
# Animated pump movement helper
# ======================================================================


def _run_animated_moves(
    s: _PumpSession,
    sock: socket.socket,
    moves: list[tuple[str, float, float]],
    live: Live,
    accel: float = 100.0,
) -> None:
    """Run one or more pump moves with real-time position animation.

    Parameters
    ----------
    moves : list of (pump_id, klipper_target, speed_mm_s)
        Pumps to move.  Multiple entries run simultaneously (sync=False).
    live : Live
        Active Rich Live context for display updates.
    accel : float
        Stepper acceleration in mm/s^2.
    """
    if not moves:
        return

    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    durations: dict[str, float] = {}

    for pid, target, speed in moves:
        starts[pid] = s.pump_positions.get(pid, 0.0)
        ends[pid] = abs(target)
        dist = abs(ends[pid] - starts[pid])
        # Overestimate by 10 % to cover accel/decel; M400 is the real sync
        durations[pid] = (dist / speed * 1.1) if speed > 0 else 0.0
        s.current_speeds[pid] = speed

    simultaneous = len(moves) > 1
    for i, (pid, target, speed) in enumerate(moves):
        is_last = (i == len(moves) - 1) and not simultaneous
        pump_move(sock, pid, target, speed, accel=accel, sync=is_last)

    max_dur = max(durations.values(), default=0.0)
    if max_dur < 0.05:
        raw_gcode(sock, "M400")
        for pid, target, speed in moves:
            s.pump_positions[pid] = abs(target)
            s.current_speeds[pid] = 0.0
        live.update(s.build_live_renderable())
        return

    t0 = time.monotonic()
    while time.monotonic() - t0 < max_dur:
        elapsed = time.monotonic() - t0
        for pid, target, speed in moves:
            dur = durations[pid]
            frac = min(1.0, elapsed / dur) if dur > 0 else 1.0
            s.pump_positions[pid] = starts[pid] + (ends[pid] - starts[pid]) * frac
        live.update(s.build_live_renderable())
        time.sleep(0.05)

    raw_gcode(sock, "M400")
    for pid, target, speed in moves:
        s.pump_positions[pid] = abs(target)
        s.current_speeds[pid] = 0.0
    live.update(s.build_live_renderable())


# ======================================================================
# Core actions (unchanged logic, output to log buffer)
# ======================================================================


def _do_homing(s: _PumpSession, pump_ids: list[str]) -> bool:
    sock = s.ensure_connected()
    backlash = s.pumps_cfg.backlash_purge_mm
    all_ok = True
    s.open_valve()
    try:
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=4
        ) as live:
            for pid in pump_ids:
                m, sy = s.motor(pid), s.syringe(pid)
                s.log(f"  Homing [bold]{s.pump_label(pid)}[/]...")
                live.update(s.build_live_renderable())
                pump_enable(sock, pid)
                ok = pump_home_with_backlash(sock, pid, m, sy, backlash)
                if ok:
                    s.homed[pid] = True
                    s.pump_positions[pid] = 0.0
                    s.log(f"  [green]{pid} homed successfully.[/]")
                else:
                    s.log(f"  [red]{pid} homing FAILED[/]")
                    all_ok = False
                pump_disable(sock, pid)
                live.update(s.build_live_renderable())
    finally:
        s.close_valve()
    s.app.session_log.log_action("pump", "homing", ", ".join(pump_ids))
    return all_ok


def _check_homing(s: _PumpSession, pump_ids: list[str]) -> bool:
    not_homed = [pid for pid in pump_ids if not s.homed[pid]]
    if not not_homed:
        return True
    labels = ", ".join(s.pump_label(pid) for pid in not_homed)
    s.log(f"  [yellow]Not homed:[/] {labels}")
    s.render_screen()
    if _ask_confirm("Home them now?"):
        return _do_homing(s, not_homed)
    s.log("  Proceeding without homing.")
    return True


def _do_dispense(s: _PumpSession, pid: str, volume_ml: float, speed: float) -> None:
    """Dispense a volume with animated progress and prompt before retract."""
    sock = s.ensure_connected()
    m, sy = s.motor(pid), s.syringe(pid)
    travel_mm = volume_to_mm(volume_ml, sy)
    if travel_mm > sy.plunger_travel_mm:
        s.log("  [yellow]Volume exceeds capacity, clamping.[/]")
        volume_ml, travel_mm = sy.volume_ml, sy.plunger_travel_mm
    dsign = _dispense_sign(m)
    pos = dsign * travel_mm
    if not s.check_travel_limit(pid, pos):
        return
    pump_enable(sock, pid)
    pump_set_position(sock, pid, 0.0)
    s.pump_positions[pid] = 0.0

    s.log(
        f"  Dispensing {volume_ml:.3f} ml ({travel_mm:.4f} mm) at {speed:.2f} mm/s..."
    )
    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=10
    ) as live:
        _run_animated_moves(s, sock, [(pid, pos, speed)], live)
    s.log("  [green]Dispense complete.[/]")
    _dwell(s)

    s.render_screen()
    if _ask_confirm("Retract plunger?"):
        retract_speed = m.max_retract_speed_mm_s
        s.log(f"  Retracting at {retract_speed:.1f} mm/s...")
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=10
        ) as live:
            _run_animated_moves(
                s, sock, [(pid, 0.0, retract_speed)], live
            )
        s.log("  [green]Retract complete.[/]")

    pump_disable(sock, pid)
    s.app.session_log.log_action("pump", "dispense", f"{pid} {volume_ml}ml @{speed}mm/s")


def _do_dispense_no_retract(
    s: _PumpSession,
    pid: str,
    volume_ml: float,
    speed: float,
    disable_after: bool = True,
) -> float:
    sock = s.ensure_connected()
    m, sy = s.motor(pid), s.syringe(pid)
    travel_mm = volume_to_mm(volume_ml, sy)
    if travel_mm > sy.plunger_travel_mm:
        volume_ml, travel_mm = sy.volume_ml, sy.plunger_travel_mm
    dsign = _dispense_sign(m)
    pos = dsign * travel_mm
    if not s.check_travel_limit(pid, pos):
        return 0.0
    pump_enable(sock, pid)
    pump_set_position(sock, pid, 0.0)
    s.pump_positions[pid] = 0.0

    s.log(
        f"  Dispensing {volume_ml:.3f} ml ({travel_mm:.4f} mm) at {speed:.2f} mm/s..."
    )
    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=10
    ) as live:
        _run_animated_moves(s, sock, [(pid, pos, speed)], live)
    s.log("  [green]Dispense complete (no retract).[/]")

    if disable_after:
        pump_disable(sock, pid)
    return travel_mm


def _do_simultaneous_dispense(
    s: _PumpSession,
    pump_ids: list[str],
    volumes: dict[str, float],
    speed: float,
    retract: bool = True,
) -> None:
    sock = s.ensure_connected()
    travels: dict[str, float] = {}
    for pid in pump_ids:
        sy = s.syringe(pid)
        t = volume_to_mm(volumes[pid], sy)
        if t > sy.plunger_travel_mm:
            t = sy.plunger_travel_mm
        m = s.motor(pid)
        dsign = _dispense_sign(m)
        if not s.check_travel_limit(pid, dsign * t):
            return
        travels[pid] = t

    max_travel = max(travels.values())
    if max_travel <= 0:
        return
    duration_s = max_travel / speed
    s.log(f"  Running pumps simultaneously ({duration_s:.1f} s)...")

    for pid in pump_ids:
        pump_enable(sock, pid)
        pump_set_position(sock, pid, 0.0)
        s.pump_positions[pid] = 0.0

    dispense_moves: list[tuple[str, float, float]] = []
    for pid in pump_ids:
        m = s.motor(pid)
        dsign = _dispense_sign(m)
        pos = dsign * travels[pid]
        pump_speed = travels[pid] / duration_s if duration_s > 0 else speed
        s.log(
            f"    {s.pump_label(pid)}: {volumes[pid]:.3f} ml at {pump_speed:.3f} mm/s"
        )
        dispense_moves.append((pid, pos, pump_speed))

    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=10
    ) as live:
        _run_animated_moves(s, sock, dispense_moves, live)

    if not retract:
        return

    _dwell(s)
    s.render_screen()
    if not _ask_confirm("Retract all plungers?"):
        return

    retract_moves: list[tuple[str, float, float]] = []
    for pid in pump_ids:
        retract_speed = travels[pid] / duration_s if duration_s > 0 else speed
        retract_moves.append((pid, 0.0, retract_speed))

    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=10
    ) as live:
        _run_animated_moves(s, sock, retract_moves, live)
    s.log("  [green]Retract complete.[/]")


# ======================================================================
# Tests
# ======================================================================


def _test_pump_juggle(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Pump Juggle ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to juggle")
    distance = _ask_float(s.app, "Distance (mm)", 1.0)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    reps = _ask_int(s.app, "Repetitions", 10)
    if not _check_homing(s, selected):
        return
    sock = s.ensure_connected()

    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=4
    ) as live:
        for pid in selected:
            m = s.motor(pid)
            dsign = _dispense_sign(m)
            pos = dsign * distance
            if not s.check_travel_limit(pid, pos):
                live.update(s.build_live_renderable())
                continue
            pump_enable(sock, pid)
            pump_set_position(sock, pid, 0.0)
            s.log(
                f"  Juggling [bold]{s.pump_label(pid)}[/]: "
                f"{distance} mm x {reps} at {speed} mm/s"
            )
            live.update(s.build_live_renderable())

            for i in range(reps):
                s.set_progress(
                    f"  Juggling {pid} [bold][{i + 1}/{reps}][/]"
                )
                _run_animated_moves(s, sock, [(pid, pos, speed)], live)
                _run_animated_moves(s, sock, [(pid, 0.0, speed)], live)

            pump_disable(sock, pid)
            s.set_progress("")
            live.update(s.build_live_renderable())

    s.log("[green]Juggle test complete.[/]")


def _test_volume(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Volume ──[/]")
    s.render_screen()
    pid = _select_single_pump(s)
    volume = _ask_float(s.app, "Volume (ml)", 0.5)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, [pid]):
        return
    _do_dispense(s, pid, volume, speed)
    s.log("[green]Volume test complete.[/]")


def _test_limit_switch(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Limit Switch ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to test")
    sock = s.ensure_connected()

    for pid in selected:
        m = s.motor(pid)
        s.log(f"  Testing [bold]{s.pump_label(pid)}[/]")
        s.log(f"  Endstop pin: {m.endstop_polarity}{m.endstop_pin}")
        s.log("  Auto-advances on OPEN -> TRIGGERED. Press Enter to skip.")

        saw_open, saw_triggered = False, False
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=3
        ) as live:
            while True:
                es = query_pump_endstop(sock, pid)
                if es.lower() == "open":
                    saw_open = True
                    s.set_progress(
                        f"  {pid}: [yellow]OPEN[/]  (press Enter to skip)"
                    )
                elif es.upper() == "TRIGGERED":
                    saw_triggered = True
                    s.set_progress(
                        f"  {pid}: [green]TRIGGERED[/]  (press Enter to skip)"
                    )
                else:
                    s.set_progress(f"  {pid}: [dim]{es}[/]  (press Enter to skip)")

                live.update(s.build_live_renderable())

                if saw_open and saw_triggered:
                    s.log(f"  [green]PASS[/]  {pid} limit switch")
                    break
                if stdin_has_data():
                    sys.stdin.readline()
                    if saw_open and not saw_triggered:
                        s.log(
                            f"  [red]FAIL[/]  {pid}: saw OPEN but never TRIGGERED"
                        )
                    elif saw_triggered and not saw_open:
                        s.log(f"  [red]FAIL[/]  {pid}: always TRIGGERED")
                    else:
                        s.log(f"  [yellow]SKIP[/]  {pid}")
                    break
                time.sleep(0.3)
        s.set_progress("")

    s.log("[green]Limit switch test complete.[/]")


def _test_speed_ramp(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Speed Ramp ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to test")
    if not _check_homing(s, selected):
        return
    sock = s.ensure_connected()
    dose_ml = 0.5

    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=4
    ) as live:
        for pid in selected:
            m, sy = s.motor(pid), s.syringe(pid)
            travel_mm = volume_to_mm(dose_ml, sy)
            dsign = _dispense_sign(m)
            dispense_pos = dsign * travel_mm
            max_spd = m.max_dispense_speed_mm_s
            speeds = [sp for sp in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0] if sp <= max_spd]
            retract_speed = m.max_retract_speed_mm_s

            if not s.check_travel_limit(pid, dispense_pos):
                live.update(s.build_live_renderable())
                continue
            s.log(
                f"  Speed ramp for [bold]{s.pump_label(pid)}[/] "
                f"({dose_ml} ml = {travel_mm:.2f} mm)"
            )
            pump_enable(sock, pid)

            for spd in speeds:
                duration_s = travel_mm / spd
                s.set_progress(
                    f"  {pid}: [cyan]{spd} mm/s[/] (~{duration_s:.1f} s)"
                )
                pump_set_position(sock, pid, 0.0)
                s.pump_positions[pid] = 0.0
                _run_animated_moves(s, sock, [(pid, dispense_pos, spd)], live)
                _dwell(s)
                _run_animated_moves(
                    s, sock, [(pid, 0.0, retract_speed)], live
                )
                _dwell(s)
                s.log(f"    [cyan]{spd} mm/s[/] [green]OK[/]")
                live.update(s.build_live_renderable())

            pump_disable(sock, pid)
            s.set_progress("")
            live.update(s.build_live_renderable())

    s.log(
        "[green]Speed ramp complete.[/] Note the highest speed with smooth, quiet motion."
    )


def _test_repeatability(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Repeatability ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to test")
    cycles = _ask_int(s.app, "Cycles", 10)
    speed = _ask_float(s.app, "Dispense speed (mm/s)", 0.5)
    if not _check_homing(s, selected):
        return
    sock = s.ensure_connected()

    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=4
    ) as live:
        for pid in selected:
            m, sy = s.motor(pid), s.syringe(pid)
            travel = sy.plunger_travel_mm
            dsign = _dispense_sign(m)
            dispense_pos = dsign * travel
            retract_speed = min(speed * 1.5, m.max_retract_speed_mm_s)
            if not s.check_travel_limit(pid, dispense_pos):
                live.update(s.build_live_renderable())
                continue

            s.log(f"  Repeatability for [bold]{s.pump_label(pid)}[/]")
            pump_enable(sock, pid)

            for i in range(cycles):
                s.set_progress(
                    f"  Cycling {pid} [bold][{i + 1}/{cycles}][/]"
                )
                _run_animated_moves(
                    s, sock, [(pid, dispense_pos, speed)], live
                )
                _dwell(s)
                _run_animated_moves(
                    s, sock, [(pid, 0.0, retract_speed)], live
                )
                if i < cycles - 1:
                    _dwell(s)
                live.update(s.build_live_renderable())

            s.set_progress("")
            s.log("  Re-homing to check for step loss...")
            live.update(s.build_live_renderable())
            ok = pump_home(sock, pid, m, sy)
            if ok:
                s.homed[pid] = True
                s.log("  [green]Re-home OK. No step loss.[/]")
            else:
                s.log("  [red]Re-home failed -- manual inspection needed.[/]")
            pump_disable(sock, pid)
            live.update(s.build_live_renderable())

    s.log("[green]Repeatability test complete.[/]")


def _test_full_travel(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Full Travel ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to test")
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, selected):
        return
    if not s.app.confirm_dangerous("Full plunger travel on selected pumps"):
        return

    sock = s.ensure_connected()
    s.console.clear()
    with Live(
        s.build_live_renderable(), console=s.console, refresh_per_second=4
    ) as live:
        for pid in selected:
            m, sy = s.motor(pid), s.syringe(pid)
            travel = sy.plunger_travel_mm
            dsign = _dispense_sign(m)
            dispense_pos = dsign * travel
            if not s.check_travel_limit(pid, dispense_pos):
                live.update(s.build_live_renderable())
                continue

            s.log(
                f"  Full travel for [bold]{s.pump_label(pid)}[/]: "
                f"{travel} mm = {sy.volume_ml} ml"
            )
            pump_enable(sock, pid)
            s.set_progress(f"  {pid}: dispensing...")
            _run_animated_moves(
                s, sock, [(pid, dispense_pos, speed)], live
            )
            s.log(f"  {pid} dispense [green]done[/]")
            _dwell(s)
            s.set_progress(f"  {pid}: retracting...")
            _run_animated_moves(s, sock, [(pid, 0.0, speed)], live)
            s.log(f"  {pid} retract [green]done[/]")
            pump_disable(sock, pid)
            s.set_progress("")
            live.update(s.build_live_renderable())

    s.log("[green]Full travel test complete.[/]")


def _test_volume_repeatability(s: _PumpSession) -> None:
    s.log("[bold cyan]── Test: Volume Repeatability ──[/]")
    s.render_screen()
    pid = _select_single_pump(s)
    volume = _ask_float(s.app, "Volume per cycle (ml)", 0.5)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, [pid]):
        return
    cycle = 0
    while True:
        cycle += 1
        s.log(f"  [bold]── Cycle {cycle} ──[/]")
        _do_dispense(s, pid, volume, speed)
        s.log("  Confirm the dispensed volume visually.")
        s.render_screen()
        if not _ask_confirm("Continue with another cycle?"):
            break
        if _ask_confirm("Change volume or speed?", default=False):
            volume = _ask_float(s.app, "Volume (ml)", volume)
            speed = _ask_float(s.app, "Speed (mm/s)", speed)
    s.log(f"[green]Volume repeatability done after {cycle} cycles.[/]")


# ======================================================================
# Control actions
# ======================================================================


def _control_homing(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Homing ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to home")
    _do_homing(s, selected)


def _control_setup(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Home + Backlash Purge ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to home + purge")
    _do_homing(s, selected)


def _control_individual_pump(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Individual Pump ──[/]")
    s.render_screen()
    pid = _select_single_pump(s)
    volume = _ask_float(s.app, "Volume (ml)", 0.5)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, [pid]):
        return
    _do_dispense_no_retract(s, pid, volume, speed)


def _control_multi_pump(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Multi-Pump ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to run")
    if not _check_homing(s, selected):
        return
    volumes: dict[str, float] = {}
    for pid in selected:
        volumes[pid] = _ask_float(s.app, f"Volume for {s.pump_label(pid)} (ml)", 0.5)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    simultaneous = _ask_confirm("Run simultaneously?", default=False)
    sock = s.ensure_connected()
    if simultaneous:
        _do_simultaneous_dispense(s, selected, volumes, speed)
        for pid in selected:
            pump_disable(sock, pid)
    else:
        s.log("  Running pumps sequentially...")
        for pid in selected:
            s.log(f"  {s.pump_label(pid)}:")
            _do_dispense(s, pid, volumes[pid], speed)


def _control_purge(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Purge ──[/]")
    purge_pid = s.purge_pump_id
    if purge_pid is None:
        s.log("  [red]No pump with fluid='purge' found.[/]")
        return
    sy = s.syringe(purge_pid)
    s.render_screen()
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, [purge_pid]):
        return
    s.log(f"  Purging full syringe ({sy.volume_ml:.2f} ml)...")
    _do_dispense(s, purge_pid, sy.volume_ml, speed)


def _control_toggle_valve(s: _PumpSession) -> None:
    """Toggle the refill valve open/closed."""
    name = s.pumps_cfg.refill_valve_output
    if not name:
        s.log("  [red]No refill_valve_output configured.[/]")
        return
    if s.valve_open_flag:
        s.close_valve()
    else:
        s.open_valve()


def _control_toggle_needle(s: _PumpSession) -> None:
    """Toggle the airbrush needle servo retracted/extended."""
    name = s.pumps_cfg.needle_output if hasattr(s.pumps_cfg, "needle_output") else None
    if not name:
        outputs = s.cfg.digital_outputs or {}
        for oname in outputs:
            if "needle" in oname.lower():
                name = oname
                break
    if not name:
        s.log("  [red]No needle output configured.[/]")
        return
    sock = s.ensure_connected()
    new_state = not s.needle_retracted
    set_pin(sock, name, 1 if new_state else 0)
    s.needle_retracted = new_state
    label = "RETRACTED" if new_state else "EXTENDED"
    s.log(f"  Needle [bold]{label}[/]")
    s.app.session_log.log_action("pump", "needle", label)


def _control_fill(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Fill ──[/]")
    s.render_screen()
    selected = _select_pumps(s, "Pumps to fill")
    speed = _ask_float(s.app, "Retract speed (mm/s)", 2.0)
    if not _check_homing(s, selected):
        return
    sock = s.ensure_connected()
    s.open_valve()
    try:
        for pid in selected:
            s.log(f"  Filling [bold]{s.pump_label(pid)}[/]...")
            pump_enable(sock, pid)

        fill_moves: list[tuple[str, float, float]] = [
            (pid, 0.0, speed) for pid in selected
        ]
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=10
        ) as live:
            _run_animated_moves(s, sock, fill_moves, live)

        for pid in selected:
            s.log(f"  {pid} fill [green]done[/]")
            pump_disable(sock, pid)
    finally:
        s.close_valve()


def _control_valve_cycle(s: _PumpSession) -> None:
    s.log("[bold cyan]── Control: Valve Cycle ──[/]")
    if not s.pumps_cfg.refill_valve_output:
        s.log("  [red]No refill_valve_output configured.[/]")
        return
    s.render_screen()
    selected = _select_pumps(s, "Pumps to cycle")
    speed = _ask_float(s.app, "Speed (mm/s)", 2.0)
    if not _check_homing(s, selected):
        return
    sock = s.ensure_connected()

    # Fill all simultaneously
    s.open_valve()
    try:
        for pid in selected:
            s.log(f"  Filling [bold]{s.pump_label(pid)}[/]...")
            pump_enable(sock, pid)

        fill_moves: list[tuple[str, float, float]] = [
            (pid, 0.0, speed) for pid in selected
        ]
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=10
        ) as live:
            _run_animated_moves(s, sock, fill_moves, live)

        for pid in selected:
            s.log(f"  {pid} fill [green]done[/]")
            pump_disable(sock, pid)
    finally:
        s.close_valve()
    _dwell(s)

    # Empty all simultaneously
    empty_moves: list[tuple[str, float, float]] = []
    for pid in selected:
        m, sy = s.motor(pid), s.syringe(pid)
        dsign = _dispense_sign(m)
        dispense_pos = dsign * sy.plunger_travel_mm
        if not s.check_travel_limit(pid, dispense_pos):
            continue
        s.log(f"  Emptying [bold]{s.pump_label(pid)}[/]...")
        pump_enable(sock, pid)
        empty_moves.append((pid, dispense_pos, speed))

    if empty_moves:
        s.console.clear()
        with Live(
            s.build_live_renderable(), console=s.console, refresh_per_second=10
        ) as live:
            _run_animated_moves(s, sock, empty_moves, live)

        for pid, _, _ in empty_moves:
            s.log(f"  {pid} empty [green]done[/]")
            pump_disable(sock, pid)

    s.log("[green]Valve cycle complete.[/]")


# ======================================================================
# Calibration
# ======================================================================


def _calibrate_volume(s: _PumpSession) -> None:
    s.log("[bold cyan]── Calibration: Volume Accuracy ──[/]")
    s.render_screen()
    pid = _select_single_pump(s, "Pump to calibrate")
    if not _check_homing(s, [pid]):
        return

    sy, m = s.syringe(pid), s.motor(pid)
    dsign = _dispense_sign(m)

    s.set_summary(
        info_panel(
            {
                "Pump": s.pump_label(pid),
                "Volume": f"{sy.volume_ml} ml",
                "Plunger travel": f"{sy.plunger_travel_mm} mm",
                "mm/ml": f"{sy.mm_per_ml:.4f}",
            },
            title="Syringe Info",
        )
    )
    s.render_screen()

    mode = questionary.select(
        "Calibration mode:",
        choices=[
            "Fixed volume -- repeat same volume N times",
            "Volume sweep -- 0.1 to 1.0 ml in 10 steps",
        ],
    ).ask()
    if mode is None:
        return

    if "sweep" in mode.lower():
        volumes = [round(0.1 * i, 1) for i in range(1, 11)]
        volumes = [v for v in volumes if v <= sy.volume_ml]
        if not volumes:
            s.log("  [red]Syringe capacity too small for sweep.[/]")
            return
        cycles = len(volumes)
    else:
        target_vol = _ask_float(s.app, "Target volume (ml)", 0.5)
        if target_vol > sy.volume_ml:
            target_vol = sy.volume_ml
        cycles = _ask_int(s.app, "Number of cycles", 5)
        volumes = [target_vol] * cycles

    speed = _ask_float(s.app, "Dispense speed (mm/s)", 0.5)

    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = data_dir / f"{pid}_volume_cal_{timestamp}.csv"
    csv_header = "cycle,target_ml,measured_ml,error_ml,error_pct,travel_mm,speed_mm_s,mm_per_ml_nominal,mm_per_ml_actual"
    csv_lines = [csv_header]

    sock = s.ensure_connected()
    results: list[dict[str, float]] = []

    s.log(f"  Starting {cycles} calibration cycle(s)")
    s.log(f"  CSV: {csv_path}")

    try:
        for i, target in enumerate(volumes):
            cycle_num = i + 1
            travel_mm = volume_to_mm(target, sy)
            pos = dsign * travel_mm

            if not s.check_travel_limit(pid, pos):
                break

            s.log(
                f"[bold]── Cycle {cycle_num}/{cycles}: "
                f"target = {target:.3f} ml ({travel_mm:.4f} mm) ──[/]"
            )

            s.open_valve()
            try:
                pump_enable(sock, pid)
                s.console.clear()
                with Live(
                    s.build_live_renderable(),
                    console=s.console,
                    refresh_per_second=10,
                ) as live:
                    _run_animated_moves(
                        s,
                        sock,
                        [(pid, 0.0, m.max_retract_speed_mm_s)],
                        live,
                    )
            finally:
                s.close_valve()

            s.render_screen()
            questionary.press_any_key_to_continue(
                "Ready to dispense. Press any key..."
            ).ask()

            pump_set_position(sock, pid, 0.0)
            s.pump_positions[pid] = 0.0
            s.log(f"  Dispensing {target:.3f} ml...")
            s.console.clear()
            with Live(
                s.build_live_renderable(),
                console=s.console,
                refresh_per_second=10,
            ) as live:
                _run_animated_moves(s, sock, [(pid, pos, speed)], live)
            s.log("  [green]Dispense complete.[/]")

            s.render_screen()
            questionary.press_any_key_to_continue(
                "Measure the output. Press any key to record..."
            ).ask()

            measured = _ask_float(s.app, "Measured volume (ml)", target)
            if measured <= 0:
                s.log("  [yellow]Must be > 0. Using target.[/]")
                measured = target

            error_ml = measured - target
            error_pct = (error_ml / target * 100.0) if target > 0 else 0.0
            actual_mm_per_ml = travel_mm / measured if measured > 0 else 0.0

            result = {
                "cycle": float(cycle_num),
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

            s.log(f"  Error: {error_ml:+.4f} ml ({error_pct:+.2f}%)")
            s.log(f"  Actual mm/ml: {actual_mm_per_ml:.4f} (nominal {sy.mm_per_ml:.4f})")

        pump_disable(sock, pid)
    except KeyboardInterrupt:
        s.log("[yellow]Calibration interrupted.[/]")
        pump_disable(sock, pid)
        if s.valve_open_flag:
            s.close_valve()

    if not results:
        return

    _show_cal_statistics(s, pid, results, csv_path)

    s.render_screen()
    _show_cal_charts(s, results)

    total_travel = sum(r["travel_mm"] for r in results)
    total_measured = sum(r["measured_ml"] for r in results)
    if total_measured > 0 and total_travel > 0:
        avg_mm_per_ml = total_travel / total_measured
        corrected_travel = avg_mm_per_ml * sy.volume_ml
        s.log(
            f"  Recommended plunger_travel_mm: [bold]{corrected_travel:.4f}[/]"
        )
        s.log(f"  (currently {sy.plunger_travel_mm:.4f})")

        s.render_screen()
        if s.app.confirm_dangerous("Save corrected value to machine.yaml"):
            _save_volume_calibration(
                {
                    pid: {
                        "plunger_travel_mm": round(corrected_travel, 4),
                        "volume_ml": sy.volume_ml,
                    }
                }
            )
            s.log("  [green]Saved. Restart to apply.[/]")


def _show_cal_statistics(
    s: _PumpSession,
    pid: str,
    results: list[dict[str, float]],
    csv_path: Path,
) -> None:
    """Compute and display calibration statistics as a summary panel."""
    n = len(results)
    errors_ml = [r["error_ml"] for r in results]
    errors_pct = [r["error_pct"] for r in results]
    measured = [r["measured_ml"] for r in results]
    targets = [r["target_ml"] for r in results]
    actual_ratios = [r["mm_per_ml_actual"] for r in results]

    mean_err = sum(errors_ml) / n
    mean_err_pct = sum(errors_pct) / n
    max_abs_err = max(abs(e) for e in errors_ml)
    mean_mm_per_ml = sum(actual_ratios) / n
    var_ml = sum((e - mean_err) ** 2 for e in errors_ml) / n
    std_ml = var_ml**0.5

    table = Table(title=f"Calibration Results: {s.pump_label(pid)}")
    table.add_column("#", justify="right")
    table.add_column("Target (ml)", justify="right")
    table.add_column("Measured (ml)", justify="right")
    table.add_column("Error (ml)", justify="right")
    table.add_column("Error %", justify="right")
    table.add_column("mm/ml", justify="right")
    for r in results:
        err_color = (
            "green"
            if abs(r["error_pct"]) < 5
            else "yellow"
            if abs(r["error_pct"]) < 10
            else "red"
        )
        table.add_row(
            f"{r['cycle']:.0f}",
            f"{r['target_ml']:.4f}",
            f"{r['measured_ml']:.4f}",
            f"[{err_color}]{r['error_ml']:+.4f}[/]",
            f"[{err_color}]{r['error_pct']:+.2f}[/]",
            f"{r['mm_per_ml_actual']:.4f}",
        )

    summary_data: dict[str, str] = {
        "Mean error": f"{mean_err:+.4f} ml ({mean_err_pct:+.2f}%)",
        "Std dev": f"{std_ml:.4f} ml",
        "Max |error|": f"{max_abs_err:.4f} ml",
        "Mean mm/ml": (
            f"{mean_mm_per_ml:.4f} (nominal {results[0]['mm_per_ml_nominal']:.4f})"
        ),
        "CSV": str(csv_path),
    }

    unique_targets = set(targets)
    if len(unique_targets) == 1:
        mean_m = sum(measured) / n
        var_rep = sum((m - mean_m) ** 2 for m in measured) / n
        std_rep = var_rep**0.5
        cv = (std_rep / mean_m * 100.0) if mean_m > 0 else 0.0
        summary_data["Repeatability CV"] = f"{cv:.2f}%"
    else:
        mean_m = sum(measured) / n
        ss_res = sum((m - t) ** 2 for m, t in zip(measured, targets))
        ss_tot = sum((m - mean_m) ** 2 for m in measured)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        summary_data["R-squared"] = f"{r_sq:.6f}"

    # Set the summary panel (replaces any previous summary)
    s.set_summary(
        Panel(
            Group(table, info_panel(summary_data, title="Summary", border_style="green")),
            title=f"[bold]Calibration: {s.pump_label(pid)}[/]",
            border_style="cyan",
        )
    )


def _show_cal_charts(s: _PumpSession, results: list[dict[str, float]]) -> None:
    """Render calibration charts using plotext (below the persistent panels)."""
    targets = [r["target_ml"] for r in results]
    measured = [r["measured_ml"] for r in results]
    errors = [r["error_ml"] for r in results]
    cycles = [r["cycle"] for r in results]

    plt.clear_figure()
    plt.theme("dark")
    plt.plot_size(60, 15)
    plt.scatter(targets, measured, label="Measured", marker="dot")
    min_v, max_v = min(targets + measured), max(targets + measured)
    plt.plot([min_v, max_v], [min_v, max_v], label="Ideal")
    plt.xlabel("Target (ml)")
    plt.ylabel("Measured (ml)")
    plt.title("Target vs Measured Volume")
    plt.show()
    s.console.print()

    plt.clear_figure()
    plt.theme("dark")
    plt.plot_size(60, 15)
    plt.bar(cycles, errors, label="Error (ml)")
    plt.xlabel("Cycle")
    plt.ylabel("Error (ml)")
    plt.title("Per-Cycle Volume Error")
    plt.show()


def _save_volume_calibration(corrections: dict[str, dict[str, float]]) -> None:
    data = load_yaml(_CFG_PATH)
    for pid, corr in corrections.items():
        motors_data = data.get("pumps", {}).get("motors", {})
        if pid not in motors_data:
            continue
        motor_data = motors_data[pid]
        if "syringe" not in motor_data:
            motor_data["syringe"] = {}
        motor_data["syringe"]["plunger_travel_mm"] = corr["plunger_travel_mm"]
        motor_data["syringe"]["volume_ml"] = corr["volume_ml"]
    atomic_yaml_dump(data, _CFG_PATH)


def _calibrate_mixing(s: _PumpSession) -> None:
    s.log("[bold cyan]── Calibration: Mixing Test ──[/]")
    s.render_screen()
    ink_ids = s.ink_pump_ids
    fluids = {pid: s.motor(pid).fluid for pid in ink_ids}

    s.log("  [bold]Ink pumps:[/]")
    for pid in ink_ids:
        s.log(f"    {s.pump_label(pid)}")

    s.log("  Enter CMY ratio (will be normalized to sum=1.0):")
    s.render_screen()

    ratios: dict[str, float] = {}
    for pid in ink_ids:
        ratios[pid] = _ask_float(
            s.app, f"Ratio for {fluids[pid]}", round(1.0 / len(ink_ids), 2)
        )

    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        s.log("  [red]Ratios must be positive.[/]")
        return
    if abs(ratio_sum - 1.0) > 0.001:
        s.log(f"  Normalizing (sum was {ratio_sum:.3f})")
        ratios = {k: v / ratio_sum for k, v in ratios.items()}

    total_vol = _ask_float(s.app, "Total volume (ml)", 1.0)
    speed = _ask_float(s.app, "Speed (mm/s)", 0.5)
    if not _check_homing(s, ink_ids):
        return

    pump_volumes = {pid: ratios[pid] * total_vol for pid in ink_ids}
    vol_table = Table(title="Per-pump volumes")
    vol_table.add_column("Pump")
    vol_table.add_column("Volume (ml)", justify="right")
    vol_table.add_column("Ratio", justify="right")
    for pid in ink_ids:
        vol_table.add_row(
            s.pump_label(pid),
            f"{pump_volumes[pid]:.3f}",
            f"{ratios[pid]:.3f}",
        )
    s.set_summary(Panel(vol_table, border_style="cyan"))
    s.render_screen()

    if not _ask_confirm("Proceed with dispensing?"):
        return

    _do_simultaneous_dispense(s, ink_ids, pump_volumes, speed, retract=False)
    for pid in ink_ids:
        pump_disable(s.ensure_connected(), pid)

    s.render_screen()
    result = questionary.text("Label for this mix (e.g. MIX_001):").ask()
    label = result if result else f"MIX_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    cmy_ratio = [ratios[pid] for pid in ink_ids]
    rgb_measured = [0.0, 0.0, 0.0]

    if _ask_confirm("Enter measured RGB now?", default=False):
        rgb_measured[0] = _ask_float(s.app, "R (0-1)", 0.0)
        rgb_measured[1] = _ask_float(s.app, "G (0-1)", 0.0)
        rgb_measured[2] = _ask_float(s.app, "B (0-1)", 0.0)

    _save_mixing_result(label, cmy_ratio, total_vol, speed, rgb_measured)
    s.log(f"  Result saved to {_MANUAL_CAL_PATH}")

    if _ask_confirm("Purge the manifold now?", default=False):
        _control_purge(s)


def _save_mixing_result(
    label: str,
    cmy_ratio: list[float],
    total_volume_ml: float,
    speed_mm_s: float,
    rgb_measured: list[float],
) -> None:
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


# ======================================================================
# Menu system
# ======================================================================

_TESTS_MENU = [
    ("Pump juggle", _test_pump_juggle),
    ("Volume test", _test_volume),
    ("Limit switch test", _test_limit_switch),
    ("Speed ramp", _test_speed_ramp),
    ("Repeatability", _test_repeatability),
    ("Full travel", _test_full_travel),
    ("Volume repeatability", _test_volume_repeatability),
]

_CONTROLS_MENU = [
    ("Homing", _control_homing),
    ("Home + Backlash purge", _control_setup),
    ("Individual pump", _control_individual_pump),
    ("Multi-pump", _control_multi_pump),
    ("Purge (manifold flush)", _control_purge),
    ("Fill (retract plunger)", _control_fill),
    ("Valve cycle (fill + empty)", _control_valve_cycle),
    ("Toggle refill valve", _control_toggle_valve),
    ("Toggle needle servo", _control_toggle_needle),
]

_CALIBRATIONS_MENU = [
    ("Volume accuracy", _calibrate_volume),
    ("Mixing test", _calibrate_mixing),
]


def _run_submenu(
    s: _PumpSession, title: str, items: list[tuple[str, Any]]
) -> None:
    """Run a persistent submenu loop.

    ``render_screen()`` is called before every prompt so the pump panel
    and activity log always occupy the top of the terminal.
    """
    while True:
        s.set_summary(None)
        s.set_progress("")
        s.render_screen()
        choices = [label for label, _ in items] + ["Back"]
        choice = questionary.select(f"{title}:", choices=choices).ask()
        if choice is None or choice == "Back":
            return

        handler = None
        for label, fn in items:
            if label == choice:
                handler = fn
                break
        if handler is None:
            continue
        try:
            handler(s)
        except KeyboardInterrupt:
            s.log("[yellow]Interrupted.[/]")
        except Exception as exc:
            s.log(f"[red]Error: {exc}[/]")
            s.app.session_log.log_error(exc)


# ======================================================================
# Entry point
# ======================================================================


def run(app: RobotApp) -> None:
    """Entry point for pump controller mode."""
    cfg = app.config
    console = app.console

    if cfg.pumps is None:
        console.print("[red]Pumps are not enabled in machine.yaml[/]")
        return

    s = _PumpSession(app)

    # Write pump-only printer.cfg and connect
    if not app.no_config_write:
        if PRINTER_CFG_PATH.exists():
            backup = PRINTER_CFG_PATH.with_suffix(".cfg.bak")
            PRINTER_CFG_PATH.rename(backup)
        config_text = generate_pump_test_cfg(cfg, pump_ids=s.pump_ids)
        PRINTER_CFG_PATH.write_text(config_text)
        s.log(f"  Wrote pump test printer.cfg to {PRINTER_CFG_PATH}")
        s.log("  Restarting Klipper...")
        restart_klipper(s.cfg.connection.socket_path)

    s.log("  Waiting for Klipper...")
    s.sock = wait_for_ready(s.cfg.connection.socket_path, timeout=45.0)
    s.log("  [green]Klipper is ready[/]")

    for pid in s.pump_ids:
        pump_disable(s.sock, pid)

    ps = s.pumps_cfg.stepper
    s.set_summary(
        info_panel(
            {
                "Pumps": str(len(s.pump_ids)),
                "Stepper": (
                    f"{ps.motor_type}, {ps.klipper_microsteps} microsteps, "
                    f"{ps.rotation_distance} mm/rev"
                ),
                "Backlash purge": f"{s.pumps_cfg.backlash_purge_mm} mm",
                "Manifold purge": f"{s.pumps_cfg.manifold_purge_volume_ml} ml",
                "Refill valve": (
                    f"{s.pumps_cfg.refill_valve_output or '(none)'} "
                    f"({s.pumps_cfg.refill_valve_delay_s:.1f}s delay)"
                ),
                "Travel limits": (
                    "ENFORCED" if s.pumps_cfg.enforce_travel_limits else "disabled"
                ),
            },
            title="Pump Controller",
            border_style="magenta",
        )
    )

    try:
        while True:
            s.render_screen()
            main_choice = questionary.select(
                "Main Menu:",
                choices=["Tests", "Control", "Calibration", "Back to main menu"],
            ).ask()

            if main_choice is None or main_choice == "Back to main menu":
                break
            elif main_choice == "Tests":
                _run_submenu(s, "Tests", _TESTS_MENU)
            elif main_choice == "Control":
                _run_submenu(s, "Control", _CONTROLS_MENU)
            elif main_choice == "Calibration":
                _run_submenu(s, "Calibration", _CALIBRATIONS_MENU)

    except KeyboardInterrupt:
        s.log("[yellow]Interrupted.[/]")
    finally:
        if s.sock is not None:
            if s.valve_open_flag:
                try:
                    s.close_valve()
                except Exception:
                    pass
            for pid in s.pump_ids:
                try:
                    pump_disable(s.sock, pid)
                except Exception:
                    pass
            s.sock.close()

    console.print("[green]Pump controller session ended.[/]")
