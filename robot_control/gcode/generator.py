"""G-code generator -- Job IR operations to G-code strings.

All coordinate transforms (canvas offset, tool offset, Y-axis flip) are
applied **here**, not at machine runtime.  The generated G-code uses
absolute machine coordinates only -- no ``G92`` commands.

Feed rate convention:
    Python stores feed rates in **mm/s**.  This module converts to the
    G-code ``F`` parameter (mm/min) at the generation boundary::

        F_value = feed_mm_s * 60.0

Direction reversal:
    High-speed direction changes can stall the stepper driver.  When a
    rapid travel reverses direction, this generator inserts a ``G4``
    dwell whose duration comes from ``steppers.direction_reversal_pause_s``
    in the machine config.
"""

from __future__ import annotations

import logging
import math
from io import StringIO
from typing import Literal

from robot_control.configs.loader import MachineConfig
from robot_control.job_ir.operations import (
    DrawPolyline,
    HomeXY,
    LinearMove,
    Operation,
    RapidXY,
    SelectTool,
    Stroke,
    ToolDown,
    ToolUp,
)

logger = logging.getLogger(__name__)


class GCodeError(Exception):
    """Raised when G-code generation fails due to invalid input."""

    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f(feed_mm_s: float) -> str:
    """Convert mm/s feed rate to G-code ``F`` parameter (mm/min)."""
    return f"F{feed_mm_s * 60.0:.1f}"


def _needs_reversal_pause(
    prev_x: float | None,
    prev_y: float | None,
    cur_x: float,
    cur_y: float,
    next_x: float,
    next_y: float,
) -> bool:
    """Detect a >90-degree direction change in XY rapids.

    Returns ``True`` when the angle between the incoming and outgoing
    travel vectors exceeds 90 degrees, which is prone to stalling the
    DM542TE driver at high speed.
    """
    if prev_x is None or prev_y is None:
        return False
    dx1 = cur_x - prev_x
    dy1 = cur_y - prev_y
    dx2 = next_x - cur_x
    dy2 = next_y - cur_y
    dot = dx1 * dx2 + dy1 * dy2
    return dot < 0  # >90 degree turn


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GCodeGenerator:
    """Convert Job IR operations to G-code.

    Parameters
    ----------
    config : MachineConfig
        Validated machine configuration.

    Notes
    -----
    Coordinate frame:
        Canvas input uses a top-left origin with +Y pointing down
        (image convention).  ``MachineConfig.canvas_to_machine()``
        applies offset + Y-flip + tool offset to produce absolute
        machine coordinates.
    """

    def __init__(self, config: MachineConfig) -> None:
        self._cfg = config
        self._tool: str = "pen"
        self._tool_is_up: bool = True
        self._last_rapid_x: float | None = None
        self._last_rapid_y: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, operations: list[Operation]) -> str:
        """Generate G-code for a flat list of operations.

        Parameters
        ----------
        operations : list[Operation]
            Job IR operations (canvas-relative, mm).

        Returns
        -------
        str
            Complete G-code program including header and footer.

        Raises
        ------
        GCodeError
            If any commanded position violates soft limits.
        """
        buf = StringIO()
        self._reset_state()
        self._write_header(buf)

        for op in operations:
            self._generate_op(op, buf)

        self._write_footer(buf)
        return buf.getvalue()

    def generate_stroke(self, stroke: Stroke) -> str:
        """Generate G-code for a single stroke (interactive mode).

        The output always ends with ``M400`` so that the Klipper API
        blocks until physical motion completes.

        Parameters
        ----------
        stroke : Stroke
            List of operations forming one atomic drawing unit.

        Returns
        -------
        str
            G-code for this stroke, terminated by ``M400``.
        """
        buf = StringIO()
        for op in stroke:
            self._generate_op(op, buf)
        buf.write("M400\n")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal: per-operation dispatch
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._tool = "pen"
        self._tool_is_up = True
        self._last_rapid_x = None
        self._last_rapid_y = None

    def _generate_op(self, op: Operation, buf: StringIO) -> None:
        if isinstance(op, HomeXY):
            self._gen_home(buf)
        elif isinstance(op, SelectTool):
            self._gen_select_tool(op, buf)
        elif isinstance(op, ToolUp):
            self._gen_tool_up(buf)
        elif isinstance(op, ToolDown):
            self._gen_tool_down(buf)
        elif isinstance(op, RapidXY):
            self._gen_rapid(op, buf)
        elif isinstance(op, LinearMove):
            self._gen_linear(op, buf)
        elif isinstance(op, DrawPolyline):
            self._gen_polyline(op, buf)
        else:
            logger.warning("Unsupported operation: %s", type(op).__name__)

    # ------------------------------------------------------------------
    # Individual generators
    # ------------------------------------------------------------------

    def _gen_home(self, buf: StringIO) -> None:
        buf.write("; --- Home XY ---\n")
        buf.write("G28 X Y\n")
        self._tool_is_up = True

    def _gen_select_tool(self, op: SelectTool, buf: StringIO) -> None:
        self._tool = op.tool
        buf.write(f"; Select tool: {op.tool}\n")
        macro = f"TOOL_{op.tool.upper()}"
        buf.write(f"{macro}\n")

    def _gen_tool_up(self, buf: StringIO) -> None:
        z = self._cfg.get_z_for_tool(self._tool, "travel")
        tc = self._cfg.get_tool(self._tool)
        buf.write(f"G0 Z{z:.3f} {_f(tc.plunge_feed_mm_s)}\n")
        self._tool_is_up = True

    def _gen_tool_down(self, buf: StringIO) -> None:
        z = self._cfg.get_z_for_tool(self._tool, "work")
        tc = self._cfg.get_tool(self._tool)
        buf.write(f"G1 Z{z:.3f} {_f(tc.plunge_feed_mm_s)}\n")
        self._tool_is_up = False

    def _gen_rapid(self, op: RapidXY, buf: StringIO) -> None:
        mx, my = self._cfg.canvas_to_machine(op.x, op.y, self._tool)
        self._validate_xy(mx, my)
        tc = self._cfg.get_tool(self._tool)

        # Direction-reversal pause
        pause_s = self._cfg.steppers.direction_reversal_pause_s
        if (
            pause_s > 0
            and self._last_rapid_x is not None
            and self._last_rapid_y is not None
        ):
            if _needs_reversal_pause(
                self._last_rapid_x,
                self._last_rapid_y,
                mx,
                my,
                mx,
                my,
            ):
                pause_ms = int(pause_s * 1000)
                buf.write(f"G4 P{pause_ms}\n")

        buf.write(f"G0 X{mx:.3f} Y{my:.3f} {_f(tc.travel_feed_mm_s)}\n")
        self._last_rapid_x = mx
        self._last_rapid_y = my

    def _gen_linear(self, op: LinearMove, buf: StringIO) -> None:
        mx, my = self._cfg.canvas_to_machine(op.x, op.y, self._tool)
        self._validate_xy(mx, my)
        tc = self._cfg.get_tool(self._tool)
        feed = op.feed if op.feed is not None else tc.feed_mm_s
        buf.write(f"G1 X{mx:.3f} Y{my:.3f} {_f(feed)}\n")

    def _gen_polyline(self, op: DrawPolyline, buf: StringIO) -> None:
        tc = self._cfg.get_tool(self._tool)
        feed = op.feed if op.feed is not None else tc.feed_mm_s

        for px, py in op.points:
            mx, my = self._cfg.canvas_to_machine(px, py, self._tool)
            self._validate_xy(mx, my)
            buf.write(f"G1 X{mx:.3f} Y{my:.3f} {_f(feed)}\n")

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------

    def _write_header(self, buf: StringIO) -> None:
        buf.write("; Generated by robot_control G-code generator\n")
        buf.write("; Units: mm, absolute positioning\n")
        buf.write("G21 ; mm mode\n")
        buf.write("G90 ; absolute positioning\n")
        buf.write("\n")

    def _write_footer(self, buf: StringIO) -> None:
        buf.write("\n")
        buf.write("; --- End of job ---\n")
        # Ensure tool is up and return to home
        z_travel = self._cfg.get_z_for_tool(self._tool, "travel")
        tc = self._cfg.get_tool(self._tool)
        buf.write(f"G0 Z{z_travel:.3f} {_f(tc.plunge_feed_mm_s)}\n")
        buf.write(f"G0 X0 Y0 {_f(tc.travel_feed_mm_s)}\n")
        buf.write("M400 ; wait for motion complete\n")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_xy(self, mx: float, my: float) -> None:
        """Reject positions outside the machine work area.

        Parameters
        ----------
        mx, my : float
            Absolute machine coordinates in mm.

        Raises
        ------
        GCodeError
            If either coordinate is out of bounds.
        """
        wa = self._cfg.work_area
        if mx < 0 or mx > wa.x:
            raise GCodeError(
                f"X={mx:.3f} mm outside work area [0, {wa.x:.1f}]"
            )
        if my < 0 or my > wa.y:
            raise GCodeError(
                f"Y={my:.3f} mm outside work area [0, {wa.y:.1f}]"
            )
