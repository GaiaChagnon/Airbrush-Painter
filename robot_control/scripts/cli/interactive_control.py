"""Interactive jog control mode -- Rich Live + readchar hybrid.

Provides real-time keyboard-driven jog control with a Rich-rendered
display.  ``readchar`` captures single keypresses without curses
taking over the terminal, while Rich ``Live`` refreshes the layout
at the configured poll interval.

Key bindings are identical to the original curses controller in
``robot_control/hardware/interactive.py``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tty
import termios
from typing import TYPE_CHECKING, Literal

import questionary
import readchar
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from robot_control.hardware.klipper_client import Position
from robot_control.scripts.cli.widgets import (
    keymap_panel,
    outputs_panel,
    position_panel,
    pump_diagram,
    render_status_bar,
)

if TYPE_CHECKING:
    from robot_control.scripts.cli.app import RobotApp

# readchar key constants
_KEY_UP = readchar.key.UP
_KEY_DOWN = readchar.key.DOWN
_KEY_LEFT = readchar.key.LEFT
_KEY_RIGHT = readchar.key.RIGHT
_KEY_PAGE_UP = readchar.key.PAGE_UP
_KEY_PAGE_DOWN = readchar.key.PAGE_DOWN
_KEY_ESC = readchar.key.ESC


_KEYMAP = {
    "Arrows": "Jog X / Y",
    "PgUp/Dn": "Jog Z",
    "+  /  -": "Change jog step",
    "H": "Home X Y Z",
    "G": "Go to position",
    "P": "Select pen",
    "A": "Select airbrush",
    "U": "Tool up",
    "L": "Tool down",
    "O": "Canvas origin",
    "1": "Pump refill servo",
    "2": "Needle servo",
    "V": "Air valve",
    "F": "Cycle pump",
    "D": "Dispense",
    "R": "Retract",
    "J": "Home pump",
    "Esc": "E-STOP",
    "Q": "Quit to menu",
}


def run(app: RobotApp) -> None:
    """Entry point for interactive control mode."""
    conn = app.connection
    cfg = app.config
    console = app.console
    log = app.session_log

    if not conn.is_connected():
        console.print("[yellow]Connecting to Klipper...[/]")
        if not app.no_config_write:
            conn.regenerate_printer_cfg(force=True)
        conn.connect()

    client = conn.client
    controller = _InteractiveJogController(app)
    controller.run()


class _InteractiveJogController:
    """Rich-rendered jog controller with readchar input."""

    def __init__(self, app: RobotApp) -> None:
        self._app = app
        self._client = app.connection.client
        self._cfg = app.config
        self._console = app.console
        self._log = app.session_log
        self._conn = app.connection

        self._tool: Literal["pen", "airbrush"] = "pen"
        self._tool_up = True
        self._homed = False
        self._status = "Ready -- press H to home"
        self._running = False

        self._output_states: dict[str, bool] = {}
        if self._cfg.digital_outputs:
            for name in self._cfg.digital_outputs:
                self._output_states[name] = False

        self._pump_ids: list[str] = []
        if self._cfg.pumps:
            self._pump_ids = list(self._cfg.pumps.motors.keys())
        self._pump_idx = 0
        self._active_pump = self._pump_ids[0] if self._pump_ids else ""
        self._pump_positions: dict[str, float] = {p: 0.0 for p in self._pump_ids}
        self._pump_homed: dict[str, bool] = {p: False for p in self._pump_ids}

        self._increments = list(self._cfg.interactive.jog_increments_mm)
        self._inc_idx = self._increments.index(
            self._cfg.interactive.default_jog_increment_mm,
        )

    @property
    def jog_increment(self) -> float:
        return self._increments[self._inc_idx]

    def run(self) -> None:
        """Main loop: Rich Live display + readchar input."""
        if not os.isatty(sys.stdin.fileno()):
            self._console.print(
                "[red]Interactive control requires a real terminal (not piped stdin).[/]"
            )
            return

        self._running = True
        fd = sys.stdin.fileno()
        self._orig_term = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=10,
                screen=True,
            ) as live:
                self._live = live
                while self._running:
                    live.update(self._build_layout())
                    key = readchar.readkey()
                    self._handle_key(key, live)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._orig_term)

    def _build_layout(self) -> Group:
        """Compose the full terminal layout."""
        pos = self._conn.get_position()

        pump_fluid = ""
        if self._active_pump and self._cfg.pumps:
            motor = self._cfg.pumps.motors.get(self._active_pump)
            if motor:
                pump_fluid = motor.fluid

        pos_p = position_panel(
            pos=pos,
            tool=self._tool,
            tool_up=self._tool_up,
            jog_step=self.jog_increment,
            homed=self._homed,
        )

        out_p = outputs_panel(
            output_states=self._output_states,
            active_pump=self._active_pump,
            pump_fluid=pump_fluid,
        )

        status_text = self._status
        status_color = "red" if status_text.startswith("ERR") else "green"
        if "!!!" in status_text:
            status_color = "bold red"
        status_p = Panel(
            Text.from_markup(f"[{status_color}]{status_text}[/]"),
            title="[bold]Status[/]",
            border_style="white",
        )

        keys_p = keymap_panel(_KEYMAP, title="Controls", columns=2)
        status_bar = render_status_bar(self._conn)

        top_row = Columns([pos_p, out_p], equal=True, expand=True)

        parts: list[object] = [top_row]
        if self._pump_ids and self._cfg.pumps:
            pump_states: dict[str, dict[str, object]] = {}
            for pid in self._pump_ids:
                motor = self._cfg.pumps.motors.get(pid)
                if motor:
                    pump_states[pid] = {
                        "fluid": motor.fluid,
                        "position_mm": abs(self._pump_positions.get(pid, 0.0)),
                        "travel_mm": motor.syringe.plunger_travel_mm,
                        "homed": self._pump_homed.get(pid, False),
                        "speed_mm_s": 0.0,
                    }
            valve_open = self._output_states.get("servo_pump_refill", False)
            needle_ret = self._output_states.get("servo_airbrush_needle", False)
            parts.append(pump_diagram(pump_states, valve_open, needle_ret))
        parts.extend([status_p, keys_p, status_bar])
        return Group(*parts)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: str, live: Live) -> None:
        if key == _KEY_ESC or key == "\x1b":
            self._emergency_stop()
        elif key in ("q", "Q"):
            self._running = False
        elif key in ("h", "H"):
            self._home()
        elif key in ("g", "G"):
            self._goto_position(live)
        elif key in ("p", "P"):
            self._select_tool("pen")
        elif key in ("a", "A"):
            self._select_tool("airbrush")
        elif key in ("u", "U"):
            self._tool_up_cmd()
        elif key in ("l", "L"):
            self._tool_down_cmd()
        elif key in ("o", "O"):
            self._goto_origin()
        elif key == "1":
            self._toggle_output("servo_pump_refill")
        elif key == "2":
            self._toggle_output("servo_airbrush_needle")
        elif key in ("v", "V"):
            self._toggle_output("air_valve")
        elif key in ("f", "F"):
            self._cycle_pump()
        elif key in ("d", "D"):
            self._pump_dispense()
        elif key in ("r", "R"):
            self._pump_retract()
        elif key in ("j", "J"):
            self._pump_home()
        elif key in ("+", "="):
            self._cycle_increment(1)
        elif key in ("-", "_"):
            self._cycle_increment(-1)
        elif key == _KEY_UP:
            self._jog(0, self.jog_increment)
        elif key == _KEY_DOWN:
            self._jog(0, -self.jog_increment)
        elif key == _KEY_RIGHT:
            self._jog(self.jog_increment, 0)
        elif key == _KEY_LEFT:
            self._jog(-self.jog_increment, 0)
        elif key == _KEY_PAGE_UP:
            self._jog_z(self.jog_increment)
        elif key == _KEY_PAGE_DOWN:
            self._jog_z(-self.jog_increment)

    # ------------------------------------------------------------------
    # Actions (mirrored from hardware/interactive.py)
    # ------------------------------------------------------------------

    def _safe_gcode(self, cmd: str, timeout: float = 15.0) -> bool:
        try:
            self._client.send_gcode(cmd, timeout=timeout)
            self._log.log_gcode(cmd)
            return True
        except Exception as exc:
            msg = str(exc)
            if len(msg) > 80:
                msg = msg[:77] + "..."
            self._status = f"ERR: {msg}"
            self._log.log_error(exc)
            return False

    def _jog(self, dx: float, dy: float) -> None:
        pos = self._conn.get_position()
        if pos is None:
            self._status = "No position -- home first"
            return
        nx, ny = pos.x + dx, pos.y + dy
        wa = self._cfg.work_area
        if nx < 0 or nx > wa.x or ny < 0 or ny > wa.y:
            self._status = f"Soft limit: X={nx:.1f} Y={ny:.1f}"
            return
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.travel_feed_mm_s * 60.0
        if self._safe_gcode(f"G91\nG0 X{dx:.3f} Y{dy:.3f} F{f_val:.1f}\nG90\nM400"):
            self._status = f"Jog X{dx:+.1f} Y{dy:+.1f}"
            self._log.log_action("interactive", "jog_xy", f"dx={dx} dy={dy}")

    def _jog_z(self, dz: float) -> None:
        pos = self._conn.get_position()
        if pos is None:
            self._status = "No position -- home first"
            return
        nz = pos.z + dz
        if nz < 0 or nz > self._cfg.work_area.z:
            self._status = f"Soft limit: Z={nz:.1f}"
            return
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        if self._safe_gcode(f"G91\nG0 Z{dz:.3f} F{f_val:.1f}\nG90\nM400"):
            self._status = f"Jog Z{dz:+.1f}"

    def _home(self) -> None:
        self._status = "Homing X Y..."
        if not self._safe_gcode("G28 X Y\nM400", timeout=60.0):
            return
        self._status = "Homing Z..."
        if self._safe_gcode("G28 Z\nM400", timeout=60.0):
            self._homed = True
            self._tool_up = True
            self._status = "Homed (X Y Z)"
            self._log.log_action("interactive", "home", "XYZ complete")

    def _select_tool(self, tool: Literal["pen", "airbrush"]) -> None:
        self._tool = tool
        macro = f"TOOL_{tool.upper()}"
        if self._safe_gcode(f"{macro}\nM400"):
            self._tool_up = True
            self._status = f"Tool: {tool}"
            self._log.log_action("interactive", "tool_select", tool)

    def _tool_up_cmd(self) -> None:
        z = self._cfg.get_z_for_tool(self._tool, "travel")
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        if self._safe_gcode(f"G0 Z{z:.3f} F{f_val:.1f}\nM400"):
            self._tool_up = True
            self._status = "Tool UP"

    def _tool_down_cmd(self) -> None:
        if not self._homed:
            self._status = "Cannot lower tool -- not homed"
            return
        z = self._cfg.get_z_for_tool(self._tool, "work")
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        if self._safe_gcode(f"G1 Z{z:.3f} F{f_val:.1f}\nM400"):
            self._tool_up = False
            self._status = "Tool DOWN"

    def _toggle_output(self, name: str) -> None:
        if not self._cfg.digital_outputs:
            self._status = "No digital outputs configured"
            return
        if name not in self._cfg.digital_outputs:
            self._status = f"Unknown output: {name}"
            return
        new_state = not self._output_states.get(name, False)
        value = 1 if new_state else 0
        if self._safe_gcode(f"SET_PIN PIN={name} VALUE={value}", timeout=5.0):
            self._output_states[name] = new_state
            label = "ON" if new_state else "OFF"
            self._status = f"{name}: {label}"
            self._log.log_action("interactive", "toggle_output", f"{name}={label}")

    def _cycle_pump(self) -> None:
        if not self._pump_ids:
            self._status = "No pumps configured"
            return
        self._pump_idx = (self._pump_idx + 1) % len(self._pump_ids)
        self._active_pump = self._pump_ids[self._pump_idx]
        fluid = ""
        if self._cfg.pumps:
            motor = self._cfg.pumps.motors.get(self._active_pump)
            if motor:
                fluid = f" ({motor.fluid})"
        self._status = f"Pump: {self._active_pump}{fluid}"

    def _pump_move(self, distance_mm: float) -> None:
        if not self._active_pump or not self._cfg.pumps:
            self._status = "No pumps configured"
            return
        motor = self._cfg.pumps.motors.get(self._active_pump)
        if not motor:
            self._status = f"Pump {self._active_pump} not in config"
            return
        speed = motor.max_dispense_speed_mm_s
        pid = self._active_pump
        # Zero the stepper position then move by distance (relative move).
        # ACCEL is required for Klipper MANUAL_STEPPER, M400 waits for
        # completion before the driver is disabled.
        cmd = (
            f"MANUAL_STEPPER STEPPER={pid} ENABLE=1\n"
            f"MANUAL_STEPPER STEPPER={pid} SET_POSITION=0\n"
            f"MANUAL_STEPPER STEPPER={pid} MOVE={distance_mm:.4f}"
            f" SPEED={speed:.4f} ACCEL=100.0\n"
            f"M400\n"
            f"MANUAL_STEPPER STEPPER={pid} ENABLE=0"
        )
        is_dispense = (distance_mm * motor.homing_direction) < 0
        direction = "dispense" if is_dispense else "retract"
        if self._safe_gcode(cmd, timeout=30.0):
            self._pump_positions[pid] = self._pump_positions.get(pid, 0.0) + distance_mm
            self._status = f"{pid} {direction} {abs(distance_mm):.3f} mm"

    def _pump_dispense(self) -> None:
        if not self._cfg.pumps:
            self._status = "No pumps configured"
            return
        motor = self._cfg.pumps.motors.get(self._active_pump)
        if not motor:
            return
        sign = -motor.homing_direction
        self._pump_move(sign * self.jog_increment)

    def _pump_retract(self) -> None:
        if not self._cfg.pumps:
            self._status = "No pumps configured"
            return
        motor = self._cfg.pumps.motors.get(self._active_pump)
        if not motor:
            return
        sign = motor.homing_direction
        self._pump_move(sign * self.jog_increment)

    def _pump_home(self) -> None:
        if not self._active_pump or not self._cfg.pumps:
            self._status = "No pumps configured"
            return
        motor = self._cfg.pumps.motors.get(self._active_pump)
        if not motor:
            return
        pid = self._active_pump
        speed = motor.homing_speed_mm_s
        h_dir = motor.homing_direction
        backlash = self._cfg.pumps.backlash_purge_mm if self._cfg.pumps else 0.3
        travel = motor.syringe.plunger_travel_mm

        self._status = f"Homing {pid}..."
        home_cmd = (
            f"MANUAL_STEPPER STEPPER={pid} ENABLE=1\n"
            f"MANUAL_STEPPER STEPPER={pid} SET_POSITION=0\n"
            f"MANUAL_STEPPER STEPPER={pid} MOVE={h_dir * (travel + 5):.4f}"
            f" SPEED={speed} ACCEL=100.0 STOP_ON_ENDSTOP=1\n"
            f"MANUAL_STEPPER STEPPER={pid} SET_POSITION=0"
        )
        if not self._safe_gcode(home_cmd, timeout=60.0):
            return

        backoff = motor.home_backoff_mm
        dsign = -h_dir
        purge_cmd = (
            f"MANUAL_STEPPER STEPPER={pid}"
            f" MOVE={-h_dir * backoff:.4f} SPEED=1.0 ACCEL=100.0\n"
            f"MANUAL_STEPPER STEPPER={pid} SET_POSITION=0\n"
            f"MANUAL_STEPPER STEPPER={pid}"
            f" MOVE={dsign * backlash:.4f} SPEED=1.0 ACCEL=100.0\n"
            f"MANUAL_STEPPER STEPPER={pid} SET_POSITION=0\n"
            f"MANUAL_STEPPER STEPPER={pid} ENABLE=0"
        )
        if self._safe_gcode(purge_cmd, timeout=60.0):
            self._pump_positions[pid] = 0.0
            self._pump_homed[pid] = True
            self._status = f"{pid} homed"
            self._log.log_action("interactive", "pump_home", pid)

    def _goto_position(self, live: Live) -> None:
        """Pause readchar, prompt with questionary, then resume."""
        live.stop()
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, self._orig_term)

        try:
            raw = questionary.text(
                "Go to (e.g. x200 y150 z10):",
            ).ask()
        finally:
            tty.setcbreak(fd)
            live.start()

        if not raw or not raw.strip():
            self._status = "Goto cancelled"
            return

        wa = self._cfg.work_area
        parts = raw.lower().replace(",", " ").replace("=", "").split()
        gcode_parts: list[str] = []
        for token in parts:
            token = token.strip()
            axis, val_str = "", ""
            if token.startswith("x"):
                axis, val_str = "X", token[1:]
            elif token.startswith("y"):
                axis, val_str = "Y", token[1:]
            elif token.startswith("z"):
                axis, val_str = "Z", token[1:]
            else:
                try:
                    float(token)
                    axis, val_str = "X", token
                except ValueError:
                    continue
            try:
                val = float(val_str)
            except ValueError:
                self._status = f"Bad value: {token}"
                return
            limit = {"X": wa.x, "Y": wa.y, "Z": wa.z}.get(axis, 0)
            if val < 0 or val > limit:
                self._status = f"Out of range: {axis}={val:.1f} (0..{limit:.0f})"
                return
            gcode_parts.append(f"{axis}{val:.3f}")

        if not gcode_parts:
            self._status = "Invalid input"
            return
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.travel_feed_mm_s * 60.0
        cmd = f"G90\nG0 {' '.join(gcode_parts)} F{f_val:.0f}\nM400"
        if self._safe_gcode(cmd, timeout=15.0):
            self._status = f"Moved to {' '.join(gcode_parts)}"
            self._log.log_action("interactive", "goto", " ".join(gcode_parts))

    def _goto_origin(self) -> None:
        if not self._homed:
            self._status = "Cannot go to origin -- not homed"
            return
        ox = self._cfg.canvas.offset_x_mm
        oy = self._cfg.canvas.offset_y_mm
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.travel_feed_mm_s * 60.0
        if self._safe_gcode(f"G0 X{ox:.3f} Y{oy:.3f} F{f_val:.1f}\nM400"):
            self._status = f"At canvas origin ({ox:.1f}, {oy:.1f})"

    def _cycle_increment(self, direction: int) -> None:
        self._inc_idx = (self._inc_idx + direction) % len(self._increments)
        self._status = f"Jog step: {self.jog_increment} mm"

    def _emergency_stop(self) -> None:
        self._conn.emergency_stop()
        self._homed = False
        self._status = "!!! EMERGENCY STOP !!!"
        self._log.log_action("interactive", "EMERGENCY_STOP")
