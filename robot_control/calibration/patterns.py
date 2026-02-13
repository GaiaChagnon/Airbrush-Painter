"""Calibration pattern generators.

Each function returns a flat ``list[Operation]`` ready for execution
via the job executor.  All dimensions are in **millimetres**, using
canvas-relative coordinates.

Common parameters accepted by every pattern:

    origin : tuple[float, float]
        Bottom-left corner of the pattern on the canvas (mm).
        Default depends on the pattern (often centred on A4 canvas).
    tool : str | None
        ``"pen"`` or ``"airbrush"``.  ``None`` keeps the current tool.
    feed : float | None
        Drawing speed override (mm/s).  ``None`` uses the tool default.
"""

from __future__ import annotations

import math
from typing import Literal

from robot_control.job_ir.operations import (
    DrawPolyline,
    HomeXY,
    LinearMove,
    Operation,
    RapidXY,
    SelectTool,
    ToolDown,
    ToolUp,
)

# Default canvas centre for A4 (210 x 297 mm) -- used when no origin given
_A4_CX = 105.0
_A4_CY = 148.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stroke(
    points: list[tuple[float, float]],
    feed: float | None = None,
) -> list[Operation]:
    """Build a standard stroke: rapid -> down -> polyline -> up."""
    if len(points) < 2:
        raise ValueError("Stroke requires at least 2 points")
    return [
        RapidXY(x=points[0][0], y=points[0][1]),
        ToolDown(),
        DrawPolyline(points=tuple(points), feed=feed),
        ToolUp(),
    ]


def _preamble(tool: str | None = None) -> list[Operation]:
    """Optional tool selection + initial ToolUp."""
    ops: list[Operation] = [ToolUp()]
    if tool is not None:
        ops.insert(0, SelectTool(tool=tool))
    return ops


# ---------------------------------------------------------------------------
# Basic geometry patterns
# ---------------------------------------------------------------------------


def square(
    size_mm: float = 50.0,
    origin: tuple[float, float] = (_A4_CX - 25.0, _A4_CY - 25.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Square pattern -- verify XY scaling.

    Parameters
    ----------
    size_mm : float
        Side length.
    """
    x0, y0 = origin
    pts = [
        (x0, y0),
        (x0 + size_mm, y0),
        (x0 + size_mm, y0 + size_mm),
        (x0, y0 + size_mm),
        (x0, y0),
    ]
    return _preamble(tool) + _stroke(pts, feed)


def rectangle(
    width: float = 60.0,
    height: float = 40.0,
    origin: tuple[float, float] = (_A4_CX - 30.0, _A4_CY - 20.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Rectangle pattern -- verify aspect ratio."""
    x0, y0 = origin
    pts = [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
        (x0, y0),
    ]
    return _preamble(tool) + _stroke(pts, feed)


def cross(
    size_mm: float = 50.0,
    origin: tuple[float, float] = (_A4_CX, _A4_CY),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Crosshair pattern -- find centre, verify axes.

    Draws two perpendicular lines centred at *origin*.
    """
    half = size_mm / 2.0
    cx, cy = origin
    h_line = [(cx - half, cy), (cx + half, cy)]
    v_line = [(cx, cy - half), (cx, cy + half)]
    return _preamble(tool) + _stroke(h_line, feed) + _stroke(v_line, feed)


def grid(
    rows: int = 5,
    cols: int = 5,
    spacing: float = 20.0,
    origin: tuple[float, float] = (30.0, 30.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Grid pattern -- overall accuracy check.

    Parameters
    ----------
    rows, cols : int
        Number of horizontal / vertical lines.
    spacing : float
        Distance between lines (mm).
    """
    ops = _preamble(tool)
    w = (cols - 1) * spacing
    h = (rows - 1) * spacing
    x0, y0 = origin

    # Horizontal lines
    for r in range(rows):
        y = y0 + r * spacing
        ops.extend(_stroke([(x0, y), (x0 + w, y)], feed))

    # Vertical lines
    for c in range(cols):
        x = x0 + c * spacing
        ops.extend(_stroke([(x, y0), (x, y0 + h)], feed))

    return ops


def diagonal(
    size_mm: float = 100.0,
    origin: tuple[float, float] = (55.0, 98.5),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Diagonal line corner-to-corner -- verify orthogonality."""
    x0, y0 = origin
    return _preamble(tool) + _stroke(
        [(x0, y0), (x0 + size_mm, y0 + size_mm)], feed,
    )


def circle(
    diameter: float = 50.0,
    segments: int = 72,
    origin: tuple[float, float] = (_A4_CX, _A4_CY),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Circle (polyline approximation) -- verify smooth motion.

    Parameters
    ----------
    diameter : float
        Circle diameter (mm).
    segments : int
        Number of line segments approximating the circle.
    """
    r = diameter / 2.0
    cx, cy = origin
    pts: list[tuple[float, float]] = []
    for i in range(segments + 1):
        angle = 2.0 * math.pi * i / segments
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return _preamble(tool) + _stroke(pts, feed)


# ---------------------------------------------------------------------------
# Calibration-specific patterns
# ---------------------------------------------------------------------------


def ruler_x(
    length: float = 100.0,
    tick_spacing: float = 10.0,
    tick_length: float = 5.0,
    origin: tuple[float, float] = (55.0, _A4_CY),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Ruler along X axis with tick marks -- measure X steps/mm.

    Parameters
    ----------
    length : float
        Total ruler length (mm).
    tick_spacing : float
        Distance between tick marks (mm).
    tick_length : float
        Length of each tick perpendicular to the ruler (mm).
    """
    x0, y0 = origin
    ops = _preamble(tool)

    # Main line
    ops.extend(_stroke([(x0, y0), (x0 + length, y0)], feed))

    # Ticks
    n_ticks = int(length / tick_spacing) + 1
    for i in range(n_ticks):
        tx = x0 + i * tick_spacing
        ops.extend(_stroke([(tx, y0 - tick_length / 2), (tx, y0 + tick_length / 2)], feed))

    return ops


def ruler_y(
    length: float = 100.0,
    tick_spacing: float = 10.0,
    tick_length: float = 5.0,
    origin: tuple[float, float] = (_A4_CX, 48.5),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Ruler along Y axis with tick marks -- measure Y steps/mm."""
    x0, y0 = origin
    ops = _preamble(tool)

    ops.extend(_stroke([(x0, y0), (x0, y0 + length)], feed))

    n_ticks = int(length / tick_spacing) + 1
    for i in range(n_ticks):
        ty = y0 + i * tick_spacing
        ops.extend(_stroke([(x0 - tick_length / 2, ty), (x0 + tick_length / 2, ty)], feed))

    return ops


def crosshair_grid(
    spacing: float = 30.0,
    arm_length: float = 5.0,
    rows: int = 3,
    cols: int = 3,
    origin: tuple[float, float] = (60.0, 103.5),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Grid of crosshairs -- tool-offset calibration.

    Parameters
    ----------
    spacing : float
        Distance between crosshair centres (mm).
    arm_length : float
        Half-length of each crosshair arm (mm).
    """
    ops = _preamble(tool)
    x0, y0 = origin
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * spacing
            cy = y0 + r * spacing
            # Horizontal arm
            ops.extend(_stroke([(cx - arm_length, cy), (cx + arm_length, cy)], feed))
            # Vertical arm
            ops.extend(_stroke([(cx, cy - arm_length), (cx, cy + arm_length)], feed))
    return ops


def speed_test(
    lengths: list[float] | None = None,
    speeds: list[float] | None = None,
    spacing: float = 10.0,
    origin: tuple[float, float] = (30.0, 50.0),
    tool: str | None = None,
) -> list[Operation]:
    """Lines at increasing speeds -- find max reliable speed.

    Parameters
    ----------
    lengths : list[float] | None
        Line lengths (mm).  Default: [50, 100, 150].
    speeds : list[float] | None
        Feed rates to test (mm/s).  Default: [10, 25, 50, 75, 100].
    spacing : float
        Vertical spacing between lines (mm).
    """
    if lengths is None:
        lengths = [50.0, 100.0, 150.0]
    if speeds is None:
        speeds = [10.0, 25.0, 50.0, 75.0, 100.0]

    ops = _preamble(tool)
    x0, y0 = origin
    row = 0
    for length in lengths:
        for speed in speeds:
            y = y0 + row * spacing
            ops.extend(_stroke([(x0, y), (x0 + length, y)], feed=speed))
            row += 1
    return ops


def acceleration_test(
    distance: float = 20.0,
    repeats: int = 10,
    origin: tuple[float, float] = (80.0, _A4_CY),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Short back-and-forth moves -- tune acceleration.

    Parameters
    ----------
    distance : float
        Length of each move (mm).
    repeats : int
        Number of back-and-forth cycles.
    """
    ops = _preamble(tool)
    x0, y0 = origin
    pts: list[tuple[float, float]] = []
    for i in range(repeats):
        if i % 2 == 0:
            pts.append((x0 + distance, y0))
        else:
            pts.append((x0, y0))
    if len(pts) >= 2:
        ops.extend(_stroke([(x0, y0)] + pts, feed))
    return ops


def backlash_test(
    distance: float = 10.0,
    repeats: int = 5,
    spacing: float = 5.0,
    origin: tuple[float, float] = (80.0, 100.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Forward/reverse pattern -- detect backlash.

    Draws overlapping lines that should coincide if there is no
    backlash.  Visible gap = backlash present.

    Parameters
    ----------
    distance : float
        Line length (mm).
    repeats : int
        Number of forward/reverse pairs.
    spacing : float
        Vertical spacing between pairs (mm).
    """
    ops = _preamble(tool)
    x0, y0 = origin
    for i in range(repeats):
        y = y0 + i * spacing
        # Forward
        ops.extend(_stroke([(x0, y), (x0 + distance, y)], feed))
        # Reverse (should overlap)
        ops.extend(_stroke([(x0 + distance, y + 0.5), (x0, y + 0.5)], feed))
    return ops


def z_touch_pattern(
    positions: list[tuple[float, float]] | None = None,
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Dots at multiple positions for Z-height calibration.

    Parameters
    ----------
    positions : list[tuple[float, float]] | None
        Canvas positions where the tool should briefly touch.
        Default: centre + four corners of a 50 mm square.
    """
    if positions is None:
        cx, cy = _A4_CX, _A4_CY
        positions = [
            (cx, cy),
            (cx - 25, cy - 25),
            (cx + 25, cy - 25),
            (cx + 25, cy + 25),
            (cx - 25, cy + 25),
        ]

    ops = _preamble(tool)
    for px, py in positions:
        ops.extend([
            RapidXY(x=px, y=py),
            ToolDown(),
            ToolUp(),
        ])
    return ops


# ---------------------------------------------------------------------------
# Pen-specific patterns
# ---------------------------------------------------------------------------


def line_weight_test(
    speeds: list[float] | None = None,
    length: float = 60.0,
    spacing: float = 8.0,
    origin: tuple[float, float] = (75.0, 80.0),
    tool: str | None = None,
) -> list[Operation]:
    """Parallel lines at different speeds -- pen pressure vs speed.

    Parameters
    ----------
    speeds : list[float] | None
        Feed rates (mm/s).  Default: [5, 10, 15, 25, 40, 60].
    """
    if speeds is None:
        speeds = [5.0, 10.0, 15.0, 25.0, 40.0, 60.0]

    ops = _preamble(tool)
    x0, y0 = origin
    for i, spd in enumerate(speeds):
        y = y0 + i * spacing
        ops.extend(_stroke([(x0, y), (x0 + length, y)], feed=spd))
    return ops


def corner_test(
    angles: list[float] | None = None,
    arm_length: float = 20.0,
    spacing: float = 30.0,
    origin: tuple[float, float] = (40.0, 80.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Zigzag patterns at different corner angles -- sharp corner quality.

    Parameters
    ----------
    angles : list[float] | None
        Corner angles in degrees.  Default: [30, 45, 60, 90, 120].
    arm_length : float
        Length of each arm (mm).
    """
    if angles is None:
        angles = [30.0, 45.0, 60.0, 90.0, 120.0]

    ops = _preamble(tool)
    x0, y0 = origin
    for i, angle_deg in enumerate(angles):
        cx = x0 + i * spacing
        # Compute two arms at the given angle
        half_rad = math.radians(angle_deg / 2.0)
        dx = arm_length * math.sin(half_rad)
        dy = arm_length * math.cos(half_rad)
        pts = [
            (cx - dx, y0 + dy),
            (cx, y0),
            (cx + dx, y0 + dy),
        ]
        ops.extend(_stroke(pts, feed))
    return ops


def fine_detail_test(
    spacings: list[float] | None = None,
    length: float = 30.0,
    num_lines: int = 5,
    origin: tuple[float, float] = (60.0, 80.0),
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Closely spaced parallel lines -- minimum feature size.

    Parameters
    ----------
    spacings : list[float] | None
        Line spacings to test (mm).  Default: [2.0, 1.0, 0.5, 0.3].
    num_lines : int
        Number of lines per group.
    """
    if spacings is None:
        spacings = [2.0, 1.0, 0.5, 0.3]

    ops = _preamble(tool)
    x0, y0 = origin
    group_gap = 15.0
    x_cursor = x0

    for sp in spacings:
        for j in range(num_lines):
            lx = x_cursor + j * sp
            ops.extend(_stroke([(lx, y0), (lx, y0 + length)], feed))
        x_cursor += num_lines * sp + group_gap

    return ops


# ---------------------------------------------------------------------------
# Composite patterns
# ---------------------------------------------------------------------------


def calibration_suite(
    tool: str | None = None,
    feed: float | None = None,
) -> list[Operation]:
    """Run the full calibration pattern suite in one job.

    Includes: square, cross, ruler_x, ruler_y, diagonal, circle.
    """
    ops: list[Operation] = []
    ops.extend(square(tool=tool, feed=feed))
    ops.extend(cross(tool=tool, feed=feed))
    ops.extend(ruler_x(tool=tool, feed=feed))
    ops.extend(ruler_y(tool=tool, feed=feed))
    ops.extend(diagonal(tool=tool, feed=feed))
    ops.extend(circle(tool=tool, feed=feed))
    return ops
