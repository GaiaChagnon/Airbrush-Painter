"""Interactive controller -- keyboard-driven manual control.

Provides a terminal UI (Python ``curses``) for jog control, tool
selection, homing, and single-stroke execution.  This is the primary
interface during machine setup and calibration.

Key map (configurable increments via ``machine.yaml``):

    Arrow keys   -- jog X / Y
    Page Up/Down -- jog Z up / down
    +/-          -- cycle jog increment
    H            -- home X and Y
    P            -- select pen tool
    A            -- select airbrush tool
    U            -- tool up (raise)
    D            -- tool down (lower)
    O            -- move to canvas origin
    Space        -- execute next stroke (step mode)
    Esc          -- EMERGENCY STOP
    Q            -- quit

Safety:
    - Every jog command checks soft limits before sending.
    - Tool-down requires homed state.
    - E-stop via Esc calls ``emergency_stop()`` (bypasses G-code queue).
    - All jog moves use ``wait=True`` (one move at a time, no pile-up).
"""

from __future__ import annotations

import curses
import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal

from robot_control.configs.loader import MachineConfig
from robot_control.hardware.klipper_client import (
    KlipperClient,
    KlipperConnectionLost,
    Position,
)

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """Snapshot of the interactive controller's runtime state."""

    position: Position | None
    tool: Literal["pen", "airbrush"]
    tool_up: bool
    jog_increment: float
    homed: bool
    status: str


class InteractiveController:
    """Terminal-based manual controller for testing and calibration.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper API client.
    config : MachineConfig
        Machine configuration.
    """

    def __init__(
        self,
        client: KlipperClient,
        config: MachineConfig,
    ) -> None:
        self._client = client
        self._cfg = config

        # Runtime state
        self._tool: Literal["pen", "airbrush"] = "pen"
        self._tool_up = True
        self._homed = False
        self._status = "Ready"
        self._position: Position | None = None
        self._running = False

        # Jog increments from config
        self._increments = list(config.interactive.jog_increments_mm)
        self._inc_idx = self._increments.index(
            config.interactive.default_jog_increment_mm,
        )

    @property
    def jog_increment(self) -> float:
        """Current jog step size in mm."""
        return self._increments[self._inc_idx]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Launch the curses TUI.  Blocks until the user quits."""
        curses.wrapper(self._main_loop)

    def _main_loop(self, stdscr: curses.window) -> None:
        """Curses event loop."""
        stdscr.nodelay(False)
        stdscr.timeout(int(self._cfg.interactive.position_poll_interval_ms))
        curses.curs_set(0)
        self._running = True

        # Start position-polling thread
        poll_stop = threading.Event()
        poll_thread = threading.Thread(
            target=self._poll_position, args=(poll_stop,), daemon=True,
        )
        poll_thread.start()

        try:
            while self._running:
                self._draw(stdscr)
                key = stdscr.getch()
                if key == -1:
                    continue
                self._handle_key(key, stdscr)
        finally:
            poll_stop.set()
            poll_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Position polling
    # ------------------------------------------------------------------

    def _poll_position(self, stop: threading.Event) -> None:
        interval = self._cfg.interactive.position_poll_interval_ms / 1000.0
        while not stop.is_set():
            try:
                self._position = self._client.get_position()
            except Exception:  # noqa: BLE001
                pass
            stop.wait(interval)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int, stdscr: curses.window) -> None:
        if key == 27:  # Esc -> E-STOP
            self._emergency_stop()
        elif key == ord("q") or key == ord("Q"):
            self._running = False
        elif key == ord("h") or key == ord("H"):
            self._home()
        elif key == ord("p") or key == ord("P"):
            self._select_tool("pen")
        elif key == ord("a") or key == ord("A"):
            self._select_tool("airbrush")
        elif key == ord("u") or key == ord("U"):
            self._tool_up_cmd()
        elif key == ord("d") or key == ord("D"):
            self._tool_down_cmd()
        elif key == ord("o") or key == ord("O"):
            self._goto_origin()
        elif key == ord("+") or key == ord("="):
            self._cycle_increment(1)
        elif key == ord("-") or key == ord("_"):
            self._cycle_increment(-1)
        elif key == curses.KEY_UP:
            self._jog(0, self.jog_increment)
        elif key == curses.KEY_DOWN:
            self._jog(0, -self.jog_increment)
        elif key == curses.KEY_RIGHT:
            self._jog(self.jog_increment, 0)
        elif key == curses.KEY_LEFT:
            self._jog(-self.jog_increment, 0)
        elif key == curses.KEY_PPAGE:  # Page Up
            self._jog_z(self.jog_increment)
        elif key == curses.KEY_NPAGE:  # Page Down
            self._jog_z(-self.jog_increment)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _jog(self, dx: float, dy: float) -> None:
        """Relative XY jog with soft-limit check."""
        if self._position is None:
            self._status = "No position -- home first"
            return
        nx = self._position.x + dx
        ny = self._position.y + dy
        wa = self._cfg.work_area
        if nx < 0 or nx > wa.x or ny < 0 or ny > wa.y:
            self._status = f"Soft limit: X={nx:.1f} Y={ny:.1f}"
            return
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.travel_feed_mm_s * 60.0
        self._client.send_gcode(
            f"G91\nG0 X{dx:.3f} Y{dy:.3f} F{f_val:.1f}\nG90\nM400",
        )
        self._status = f"Jog X{dx:+.1f} Y{dy:+.1f}"

    def _jog_z(self, dz: float) -> None:
        """Relative Z jog with soft-limit check."""
        if self._position is None:
            self._status = "No position -- home first"
            return
        nz = self._position.z + dz
        if nz < 0 or nz > self._cfg.work_area.z:
            self._status = f"Soft limit: Z={nz:.1f}"
            return
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        self._client.send_gcode(
            f"G91\nG0 Z{dz:.3f} F{f_val:.1f}\nG90\nM400",
        )
        self._status = f"Jog Z{dz:+.1f}"

    def _home(self) -> None:
        self._status = "Homing X Y..."
        try:
            self._client.send_gcode("G28 X Y\nM400", timeout=30.0)
            self._homed = True
            self._tool_up = True
            self._status = "Homed"
        except Exception as exc:  # noqa: BLE001
            self._status = f"Home failed: {exc}"

    def _select_tool(self, tool: Literal["pen", "airbrush"]) -> None:
        self._tool = tool
        macro = f"TOOL_{tool.upper()}"
        try:
            self._client.send_gcode(f"{macro}\nM400")
            self._tool_up = True
            self._status = f"Tool: {tool}"
        except Exception as exc:  # noqa: BLE001
            self._status = f"Tool select failed: {exc}"

    def _tool_up_cmd(self) -> None:
        z = self._cfg.get_z_for_tool(self._tool, "travel")
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        self._client.send_gcode(f"G0 Z{z:.3f} F{f_val:.1f}\nM400")
        self._tool_up = True
        self._status = "Tool UP"

    def _tool_down_cmd(self) -> None:
        if not self._homed:
            self._status = "Cannot lower tool -- not homed"
            return
        z = self._cfg.get_z_for_tool(self._tool, "work")
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.plunge_feed_mm_s * 60.0
        self._client.send_gcode(f"G1 Z{z:.3f} F{f_val:.1f}\nM400")
        self._tool_up = False
        self._status = "Tool DOWN"

    def _goto_origin(self) -> None:
        if not self._homed:
            self._status = "Cannot go to origin -- not homed"
            return
        ox = self._cfg.canvas.offset_x_mm
        oy = self._cfg.canvas.offset_y_mm
        tc = self._cfg.get_tool(self._tool)
        f_val = tc.travel_feed_mm_s * 60.0
        self._client.send_gcode(
            f"G0 X{ox:.3f} Y{oy:.3f} F{f_val:.1f}\nM400",
        )
        self._status = f"At canvas origin ({ox:.1f}, {oy:.1f})"

    def _cycle_increment(self, direction: int) -> None:
        self._inc_idx = (self._inc_idx + direction) % len(self._increments)
        self._status = f"Jog step: {self.jog_increment} mm"

    def _emergency_stop(self) -> None:
        self._client.emergency_stop()
        self._homed = False
        self._status = "!!! EMERGENCY STOP !!!"
        logger.warning("E-STOP triggered from interactive controller")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _draw(self, stdscr: curses.window) -> None:
        """Render the terminal UI."""
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        cw = min(w - 2, 60)  # content width

        pos = self._position
        px = f"{pos.x:.3f}" if pos else "---"
        py = f"{pos.y:.3f}" if pos else "---"
        pz = f"{pos.z:.3f}" if pos else "---"

        tool_state = "up" if self._tool_up else "DOWN"

        lines = [
            f"{'INTERACTIVE CONTROL':^{cw}}",
            "-" * cw,
            f"  Position:  X: {px:>10}   Y: {py:>10}   Z: {pz:>10}",
            f"  Tool:      {self._tool.upper()} ({tool_state})",
            f"  Jog Step:  {self.jog_increment} mm",
            f"  State:     {self._status}",
            "-" * cw,
            "  [Arrows] Jog XY   [PgUp/Dn] Jog Z   [+/-] Step size",
            "  [H] Home          [P] Pen       [A] Airbrush",
            "  [U] Up            [D] Down      [O] Canvas origin",
            "  [Esc] E-STOP      [Q] Quit",
        ]

        for row, line in enumerate(lines):
            if row >= h - 1:
                break
            stdscr.addnstr(row, 1, line, w - 2)

        stdscr.refresh()
