"""
Interactive Controller.

Provides keyboard-driven manual control for testing and calibration.
Terminal UI using Python curses (Linux native).
"""

from __future__ import annotations

import curses
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

from robot_control.configs.loader import MachineConfig
from robot_control.hardware.klipper_client import KlipperClient, Position

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """Current interactive controller state."""

    position: Position | None
    tool: Literal["pen", "airbrush"]
    tool_up: bool
    jog_increment: float
    homed: bool
    status: str


class InteractiveController:
    """
    Keyboard-driven manual control for testing and calibration.

    Provides jog control, tool selection, and real-time position display
    in a terminal UI.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper API client.
    config : MachineConfig
        Machine configuration.

    Examples
    --------
    >>> from robot_control.hardware.klipper_client import KlipperClient
    >>> from robot_control.configs.loader import load_config
    >>> config = load_config()
    >>> with KlipperClient(config.connection.socket_path) as client:
    ...     controller = InteractiveController(client, config)
    ...     controller.run()  # Enters interactive mode
    """

    # Key codes
    KEY_UP = curses.KEY_UP
    KEY_DOWN = curses.KEY_DOWN
    KEY_LEFT = curses.KEY_LEFT
    KEY_RIGHT = curses.KEY_RIGHT
    KEY_PAGEUP = curses.KEY_PPAGE
    KEY_PAGEDOWN = curses.KEY_NPAGE

    def __init__(self, client: KlipperClient, config: MachineConfig) -> None:
        self.client = client
        self.config = config

        # State
        self._tool: Literal["pen", "airbrush"] = "pen"
        self._tool_up = True
        self._jog_index = config.interactive.jog_increments_mm.index(
            config.interactive.default_jog_increment_mm
        )
        self._position: Position | None = None
        self._homed = False
        self._status = "Ready"
        self._running = False

        # Threading
        self._position_lock = threading.Lock()
        self._position_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def jog_increment(self) -> float:
        """Current jog increment in mm."""
        return self.config.interactive.jog_increments_mm[self._jog_index]

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        with self._position_lock:
            return ControllerState(
                position=self._position,
                tool=self._tool,
                tool_up=self._tool_up,
                jog_increment=self.jog_increment,
                homed=self._homed,
                status=self._status,
            )

    def run(self) -> None:
        """
        Enter interactive control mode.

        Runs curses-based terminal UI until user quits (Q key).
        """
        self._running = True
        self._stop_event.clear()

        # Start position polling thread
        self._position_thread = threading.Thread(
            target=self._position_poll_loop,
            daemon=True,
        )
        self._position_thread.start()

        # Check initial homing state
        try:
            self._homed = self.client.is_homed("xy")
        except Exception:
            self._homed = False

        # Run curses UI
        try:
            curses.wrapper(self._curses_main)
        finally:
            self._running = False
            self._stop_event.set()
            if self._position_thread:
                self._position_thread.join(timeout=1.0)

    def _curses_main(self, stdscr: curses.window) -> None:
        """Main curses event loop."""
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(50)  # 50ms refresh rate

        # Initialize color pairs if available
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

        while self._running:
            # Handle input
            key = stdscr.getch()
            if key != -1:
                if not self._handle_key(key):
                    break  # Quit requested

            # Redraw display
            self._draw_display(stdscr)

    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.

        Returns False to quit, True to continue.
        """
        try:
            # Quit
            if key in (ord("q"), ord("Q")):
                return False

            # Emergency stop
            if key == 27:  # Escape
                self._emergency_stop()
                return True

            # Movement keys
            if key == self.KEY_LEFT:
                self._jog("X", -1)
            elif key == self.KEY_RIGHT:
                self._jog("X", 1)
            elif key == self.KEY_UP:
                self._jog("Y", 1)
            elif key == self.KEY_DOWN:
                self._jog("Y", -1)
            elif key == self.KEY_PAGEUP:
                self._jog("Z", 1)
            elif key == self.KEY_PAGEDOWN:
                self._jog("Z", -1)

            # Jog increment
            elif key in (ord("+"), ord("=")):
                self._change_jog_increment(1)
            elif key in (ord("-"), ord("_")):
                self._change_jog_increment(-1)

            # Homing
            elif key in (ord("h"), ord("H")):
                self._home()

            # Tool selection
            elif key in (ord("p"), ord("P")):
                self._select_tool("pen")
            elif key in (ord("a"), ord("A")):
                self._select_tool("airbrush")

            # Tool up/down
            elif key in (ord("u"), ord("U")):
                self._tool_up_cmd()
            elif key in (ord("d"), ord("D")):
                self._tool_down_cmd()

            # Go to origin
            elif key in (ord("o"), ord("O")):
                self._go_to_origin()

        except Exception as e:
            self._status = f"Error: {e}"
            logger.error("Key handler error: %s", e)

        return True

    def _jog(self, axis: str, direction: int) -> None:
        """Execute jog move on specified axis."""
        if not self._homed and axis in ("X", "Y"):
            self._status = "Home required before XY jog"
            return

        distance = self.jog_increment * direction
        feed = self.config.get_tool(self._tool).travel_feed_mm_min

        # Build G-code
        gcode = f"G91\nG0 {axis}{distance:.3f} F{feed:.0f}\nG90\nM400"

        self._status = f"Jogging {axis} {'+' if direction > 0 else ''}{distance:.3f}mm"
        try:
            self.client.send_gcode(gcode)
        except Exception as e:
            self._status = f"Jog failed: {e}"

    def _change_jog_increment(self, delta: int) -> None:
        """Change jog increment by index delta."""
        new_index = self._jog_index + delta
        if 0 <= new_index < len(self.config.interactive.jog_increments_mm):
            self._jog_index = new_index
            self._status = f"Jog increment: {self.jog_increment}mm"

    def _home(self) -> None:
        """Home X and Y axes."""
        self._status = "Homing X and Y..."
        try:
            self.client.send_gcode("G28 X Y\nM400")
            self._homed = True
            self._status = "Homed"
        except Exception as e:
            self._status = f"Home failed: {e}"
            self._homed = False

    def _select_tool(self, tool: Literal["pen", "airbrush"]) -> None:
        """Select active tool."""
        self._tool = tool
        macro = "TOOL_PEN" if tool == "pen" else "TOOL_AIRBRUSH"
        self._status = f"Selected {tool}"
        try:
            self.client.send_gcode(f"{macro}\nM400")
        except Exception as e:
            self._status = f"Tool select failed: {e}"

    def _tool_up_cmd(self) -> None:
        """Raise tool to travel height."""
        z = self.config.z_states.travel_mm
        feed = self.config.get_tool(self._tool).plunge_feed_mm_min
        self._status = "Raising tool..."
        try:
            self.client.send_gcode(f"G0 Z{z:.3f} F{feed:.0f}\nM400")
            self._tool_up = True
            self._status = "Tool up"
        except Exception as e:
            self._status = f"Tool up failed: {e}"

    def _tool_down_cmd(self) -> None:
        """Lower tool to work height."""
        if not self._homed:
            self._status = "Home required before tool down"
            return

        z = self.config.get_z_for_tool(self._tool, "work")
        feed = self.config.get_tool(self._tool).plunge_feed_mm_min
        self._status = "Lowering tool..."
        try:
            self.client.send_gcode(f"G1 Z{z:.3f} F{feed:.0f}\nM400")
            self._tool_up = False
            self._status = f"Tool down ({self._tool})"
        except Exception as e:
            self._status = f"Tool down failed: {e}"

    def _go_to_origin(self) -> None:
        """Move to canvas origin (0, 0)."""
        if not self._homed:
            self._status = "Home required"
            return

        # Ensure tool is up
        self._tool_up_cmd()

        mx, my = self.config.canvas_to_machine(0, 0, self._tool)
        feed = self.config.get_tool(self._tool).travel_feed_mm_min
        self._status = "Moving to canvas origin..."
        try:
            self.client.send_gcode(f"G0 X{mx:.3f} Y{my:.3f} F{feed:.0f}\nM400")
            self._status = "At canvas origin"
        except Exception as e:
            self._status = f"Move failed: {e}"

    def _emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._status = "EMERGENCY STOP!"
        try:
            self.client.emergency_stop()
        except Exception:
            pass
        self._homed = False

    def _position_poll_loop(self) -> None:
        """Background thread for position updates."""
        interval = self.config.interactive.position_poll_interval_ms / 1000.0

        while not self._stop_event.is_set():
            try:
                pos = self.client.get_position()
                with self._position_lock:
                    self._position = pos
            except Exception as e:
                logger.debug("Position poll error: %s", e)

            self._stop_event.wait(interval)

    def _draw_display(self, stdscr: curses.window) -> None:
        """Draw the terminal UI."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        self._draw_header(stdscr, width)

        # Position display
        self._draw_position(stdscr, 3)

        # State display
        self._draw_state(stdscr, 8)

        # Help
        self._draw_help(stdscr, 14, height)

        stdscr.refresh()

    def _draw_header(self, stdscr: curses.window, width: int) -> None:
        """Draw header bar."""
        header = " INTERACTIVE CONTROL "
        padding = (width - len(header)) // 2

        if curses.has_colors():
            stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(0, 0, "═" * width)
        stdscr.addstr(1, padding, header)
        stdscr.addstr(2, 0, "═" * width)
        if curses.has_colors():
            stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)

    def _draw_position(self, stdscr: curses.window, start_row: int) -> None:
        """Draw position display."""
        with self._position_lock:
            pos = self._position

        stdscr.addstr(start_row, 2, "Position:", curses.A_BOLD)

        if pos:
            stdscr.addstr(start_row + 1, 4, f"X: {pos.x:8.3f} mm")
            stdscr.addstr(start_row + 2, 4, f"Y: {pos.y:8.3f} mm")
            stdscr.addstr(start_row + 3, 4, f"Z: {pos.z:8.3f} mm")
        else:
            stdscr.addstr(start_row + 1, 4, "X: ---")
            stdscr.addstr(start_row + 2, 4, "Y: ---")
            stdscr.addstr(start_row + 3, 4, "Z: ---")

    def _draw_state(self, stdscr: curses.window, start_row: int) -> None:
        """Draw state display."""
        stdscr.addstr(start_row, 2, "State:", curses.A_BOLD)

        # Tool
        tool_str = f"Tool: {self._tool.upper()}"
        if self._tool_up:
            tool_str += " (up)"
        else:
            tool_str += " (down)"
        stdscr.addstr(start_row + 1, 4, tool_str)

        # Jog increment
        stdscr.addstr(start_row + 2, 4, f"Jog Step: {self.jog_increment} mm")

        # Homed status
        if self._homed:
            if curses.has_colors():
                stdscr.attron(curses.color_pair(1))
            stdscr.addstr(start_row + 3, 4, "Homed: YES")
            if curses.has_colors():
                stdscr.attroff(curses.color_pair(1))
        else:
            if curses.has_colors():
                stdscr.attron(curses.color_pair(2))
            stdscr.addstr(start_row + 3, 4, "Homed: NO")
            if curses.has_colors():
                stdscr.attroff(curses.color_pair(2))

        # Status message
        if curses.has_colors():
            stdscr.attron(curses.color_pair(3))
        stdscr.addstr(start_row + 4, 4, f"Status: {self._status}")
        if curses.has_colors():
            stdscr.attroff(curses.color_pair(3))

    def _draw_help(self, stdscr: curses.window, start_row: int, max_height: int) -> None:
        """Draw help text."""
        if start_row >= max_height - 8:
            return

        stdscr.addstr(start_row, 2, "Controls:", curses.A_BOLD)
        help_lines = [
            "[←→↑↓] Jog XY    [PgUp/Dn] Jog Z    [+/-] Step size",
            "[H] Home XY      [O] Go to origin",
            "[P] Pen tool     [A] Airbrush tool",
            "[U] Tool up      [D] Tool down",
            "[Esc] E-STOP     [Q] Quit",
        ]

        for i, line in enumerate(help_lines):
            row = start_row + 1 + i
            if row < max_height - 1:
                stdscr.addstr(row, 4, line)


def run_interactive(
    socket_path: str | None = None,
    config_path: str | None = None,
) -> None:
    """
    Convenience function to run interactive control.

    Parameters
    ----------
    socket_path : str | None
        Klipper socket path. Uses config default if None.
    config_path : str | None
        Config file path. Uses default if None.
    """
    from robot_control.configs.loader import load_config

    config = load_config(config_path)

    if socket_path is None:
        socket_path = config.connection.socket_path

    with KlipperClient(
        socket_path,
        timeout=config.connection.timeout_s,
        reconnect_attempts=config.connection.reconnect_attempts,
    ) as client:
        controller = InteractiveController(client, config)
        controller.run()
