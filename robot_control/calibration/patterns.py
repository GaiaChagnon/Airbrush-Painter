"""
Calibration Pattern Generators.

Generate test patterns for calibration and verification. Each pattern is a
function that returns Job IR operations.

All dimensions are in millimeters, canvas-relative coordinates.
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


def _stroke(
    points: list[tuple[float, float]],
    feed: float | None = None,
) -> list[Operation]:
    """
    Create a complete stroke from polyline points.

    Rapid to first point, lower tool, draw polyline, raise tool.
    """
    if len(points) < 2:
        raise ValueError("Stroke requires at least 2 points")

    return [
        RapidXY(x=points[0][0], y=points[0][1]),
        ToolDown(),
        DrawPolyline(points=tuple(points), feed=feed),
        ToolUp(),
    ]


# --- Basic Geometry Patterns ---


def square(
    size_mm: float = 50.0,
    origin: tuple[float, float] = (80.0, 120.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a square.

    Parameters
    ----------
    size_mm : float
        Side length in mm.
    origin : tuple[float, float]
        Bottom-left corner position (x, y) on canvas.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the square.
    """
    x, y = origin
    points = [
        (x, y),
        (x + size_mm, y),
        (x + size_mm, y + size_mm),
        (x, y + size_mm),
        (x, y),  # Close the square
    ]
    return [ToolUp()] + _stroke(points, feed)


def rectangle(
    width_mm: float = 80.0,
    height_mm: float = 60.0,
    origin: tuple[float, float] = (60.0, 100.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a rectangle.

    Parameters
    ----------
    width_mm : float
        Rectangle width in mm.
    height_mm : float
        Rectangle height in mm.
    origin : tuple[float, float]
        Bottom-left corner position on canvas.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the rectangle.
    """
    x, y = origin
    points = [
        (x, y),
        (x + width_mm, y),
        (x + width_mm, y + height_mm),
        (x, y + height_mm),
        (x, y),
    ]
    return [ToolUp()] + _stroke(points, feed)


def cross(
    size_mm: float = 50.0,
    center: tuple[float, float] = (105.0, 148.5),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a cross (plus sign).

    Useful for finding center and verifying axis alignment.

    Parameters
    ----------
    size_mm : float
        Total length of each arm.
    center : tuple[float, float]
        Center position on canvas.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the cross.
    """
    cx, cy = center
    half = size_mm / 2

    # Horizontal line
    h_points = [(cx - half, cy), (cx + half, cy)]
    # Vertical line
    v_points = [(cx, cy - half), (cx, cy + half)]

    return [ToolUp()] + _stroke(h_points, feed) + _stroke(v_points, feed)


def grid(
    rows: int = 5,
    cols: int = 5,
    spacing_mm: float = 20.0,
    origin: tuple[float, float] = (30.0, 50.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a grid pattern.

    Parameters
    ----------
    rows : int
        Number of horizontal lines.
    cols : int
        Number of vertical lines.
    spacing_mm : float
        Distance between lines.
    origin : tuple[float, float]
        Bottom-left corner position.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the grid.
    """
    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin
    width = (cols - 1) * spacing_mm
    height = (rows - 1) * spacing_mm

    # Horizontal lines
    for i in range(rows):
        y = y0 + i * spacing_mm
        ops.extend(_stroke([(x0, y), (x0 + width, y)], feed))

    # Vertical lines
    for j in range(cols):
        x = x0 + j * spacing_mm
        ops.extend(_stroke([(x, y0), (x, y0 + height)], feed))

    return ops


def diagonal(
    size_mm: float = 50.0,
    origin: tuple[float, float] = (80.0, 120.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a diagonal line from corner to corner of a square.

    Useful for verifying orthogonality.

    Parameters
    ----------
    size_mm : float
        Size of the bounding square.
    origin : tuple[float, float]
        Bottom-left corner position.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the diagonal.
    """
    x, y = origin
    points = [(x, y), (x + size_mm, y + size_mm)]
    return [ToolUp()] + _stroke(points, feed)


def circle(
    diameter_mm: float = 50.0,
    center: tuple[float, float] = (105.0, 148.5),
    segments: int = 64,
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a circle approximated by line segments.

    Parameters
    ----------
    diameter_mm : float
        Circle diameter in mm.
    center : tuple[float, float]
        Center position on canvas.
    segments : int
        Number of line segments (more = smoother).
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the circle.
    """
    cx, cy = center
    radius = diameter_mm / 2

    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))

    return [ToolUp()] + _stroke(points, feed)


# --- Calibration-Specific Patterns ---


def ruler_x(
    length_mm: float = 100.0,
    tick_spacing_mm: float = 10.0,
    tick_height_mm: float = 5.0,
    origin: tuple[float, float] = (50.0, 148.5),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a horizontal ruler with tick marks.

    Used for calibrating X axis steps/mm.

    Parameters
    ----------
    length_mm : float
        Total ruler length.
    tick_spacing_mm : float
        Distance between tick marks.
    tick_height_mm : float
        Height of tick marks.
    origin : tuple[float, float]
        Left end of ruler.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the ruler.
    """
    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    # Main horizontal line
    ops.extend(_stroke([(x0, y0), (x0 + length_mm, y0)], feed))

    # Tick marks
    num_ticks = int(length_mm / tick_spacing_mm) + 1
    for i in range(num_ticks):
        x = x0 + i * tick_spacing_mm
        # Taller tick every 50mm
        h = tick_height_mm * 1.5 if (i * tick_spacing_mm) % 50 == 0 else tick_height_mm
        ops.extend(_stroke([(x, y0), (x, y0 + h)], feed))

    return ops


def ruler_y(
    length_mm: float = 100.0,
    tick_spacing_mm: float = 10.0,
    tick_height_mm: float = 5.0,
    origin: tuple[float, float] = (105.0, 50.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a vertical ruler with tick marks.

    Used for calibrating Y axis steps/mm.

    Parameters
    ----------
    length_mm : float
        Total ruler length.
    tick_spacing_mm : float
        Distance between tick marks.
    tick_height_mm : float
        Width of tick marks.
    origin : tuple[float, float]
        Bottom end of ruler.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the ruler.
    """
    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    # Main vertical line
    ops.extend(_stroke([(x0, y0), (x0, y0 + length_mm)], feed))

    # Tick marks
    num_ticks = int(length_mm / tick_spacing_mm) + 1
    for i in range(num_ticks):
        y = y0 + i * tick_spacing_mm
        # Taller tick every 50mm
        w = tick_height_mm * 1.5 if (i * tick_spacing_mm) % 50 == 0 else tick_height_mm
        ops.extend(_stroke([(x0, y), (x0 + w, y)], feed))

    return ops


def crosshair_grid(
    rows: int = 3,
    cols: int = 3,
    spacing_mm: float = 50.0,
    crosshair_size_mm: float = 10.0,
    origin: tuple[float, float] = (50.0, 80.0),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw a grid of crosshairs.

    Used for tool offset calibration - draw with one tool, then overlay
    with the other to measure offset.

    Parameters
    ----------
    rows : int
        Number of crosshair rows.
    cols : int
        Number of crosshair columns.
    spacing_mm : float
        Distance between crosshairs.
    crosshair_size_mm : float
        Total size of each crosshair.
    origin : tuple[float, float]
        Position of bottom-left crosshair.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw the crosshair grid.
    """
    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin
    half = crosshair_size_mm / 2

    for i in range(rows):
        for j in range(cols):
            cx = x0 + j * spacing_mm
            cy = y0 + i * spacing_mm

            # Horizontal line
            ops.extend(_stroke([(cx - half, cy), (cx + half, cy)], feed))
            # Vertical line
            ops.extend(_stroke([(cx, cy - half), (cx, cy + half)], feed))

    return ops


def speed_test(
    lengths_mm: list[float] | None = None,
    speeds_mm_min: list[float] | None = None,
    origin: tuple[float, float] = (30.0, 50.0),
    line_spacing_mm: float = 10.0,
) -> list[Operation]:
    """
    Draw lines at different speeds to find optimal feed rate.

    Parameters
    ----------
    lengths_mm : list[float] | None
        Line lengths to test. Default: [50, 100, 150].
    speeds_mm_min : list[float] | None
        Feed rates to test. Default: [500, 1000, 1500, 2000, 3000].
    origin : tuple[float, float]
        Starting position for first line.
    line_spacing_mm : float
        Vertical spacing between lines.

    Returns
    -------
    list[Operation]
        Job IR operations to draw speed test pattern.
    """
    if lengths_mm is None:
        lengths_mm = [50.0, 100.0, 150.0]
    if speeds_mm_min is None:
        speeds_mm_min = [500.0, 1000.0, 1500.0, 2000.0, 3000.0]

    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    row = 0
    for length in lengths_mm:
        for speed in speeds_mm_min:
            y = y0 + row * line_spacing_mm
            ops.extend(_stroke([(x0, y), (x0 + length, y)], feed=speed))
            row += 1

    return ops


def backlash_test(
    distance_mm: float = 20.0,
    repeats: int = 5,
    origin: tuple[float, float] = (105.0, 148.5),
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw forward/reverse pattern to detect backlash.

    If backlash exists, the overlapping lines will show visible gaps
    or offsets on direction reversals.

    Parameters
    ----------
    distance_mm : float
        Distance to travel in each direction.
    repeats : int
        Number of forward/reverse cycles.
    origin : tuple[float, float]
        Starting position.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw backlash test.
    """
    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    # Draw horizontal forward/reverse pattern
    points: list[tuple[float, float]] = [(x0, y0)]
    for i in range(repeats):
        if i % 2 == 0:
            points.append((x0 + distance_mm, y0))
        else:
            points.append((x0, y0))

    ops.extend(_stroke(points, feed))

    # Draw vertical forward/reverse pattern
    v_points: list[tuple[float, float]] = [(x0 + distance_mm + 10, y0)]
    for i in range(repeats):
        if i % 2 == 0:
            v_points.append((x0 + distance_mm + 10, y0 + distance_mm))
        else:
            v_points.append((x0 + distance_mm + 10, y0))

    ops.extend(_stroke(v_points, feed))

    return ops


def z_touch_pattern(
    positions: list[tuple[float, float]] | None = None,
    dot_size_mm: float = 2.0,
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw small marks at specified positions for Z calibration.

    Used to verify pen touches paper at each position.

    Parameters
    ----------
    positions : list[tuple[float, float]] | None
        Canvas positions to mark. Default: 5x3 grid.
    dot_size_mm : float
        Size of each mark.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations to draw Z touch pattern.
    """
    if positions is None:
        # Default 5x3 grid across canvas
        positions = [
            (30, 50), (105, 50), (180, 50),
            (30, 148), (105, 148), (180, 148),
            (30, 247), (105, 247), (180, 247),
        ]

    ops: list[Operation] = [ToolUp()]

    for x, y in positions:
        # Draw small cross at each position
        half = dot_size_mm / 2
        ops.extend(_stroke([(x - half, y), (x + half, y)], feed))
        ops.extend(_stroke([(x, y - half), (x, y + half)], feed))

    return ops


# --- Pen-Specific Patterns ---


def line_weight_test(
    speeds_mm_min: list[float] | None = None,
    line_length_mm: float = 80.0,
    origin: tuple[float, float] = (60.0, 50.0),
    line_spacing_mm: float = 8.0,
) -> list[Operation]:
    """
    Draw parallel lines at different speeds to test line weight.

    Slower speeds typically produce thicker lines with pen.

    Parameters
    ----------
    speeds_mm_min : list[float] | None
        Feed rates to test. Default: [300, 500, 800, 1000, 1500, 2000].
    line_length_mm : float
        Length of each test line.
    origin : tuple[float, float]
        Starting position.
    line_spacing_mm : float
        Vertical spacing between lines.

    Returns
    -------
    list[Operation]
        Job IR operations for line weight test.
    """
    if speeds_mm_min is None:
        speeds_mm_min = [300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0]

    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    for i, speed in enumerate(speeds_mm_min):
        y = y0 + i * line_spacing_mm
        ops.extend(_stroke([(x0, y), (x0 + line_length_mm, y)], feed=speed))

    return ops


def corner_test(
    angles: list[float] | None = None,
    arm_length_mm: float = 30.0,
    origin: tuple[float, float] = (50.0, 50.0),
    spacing_mm: float = 40.0,
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw corners at various angles to test sharp corner quality.

    Parameters
    ----------
    angles : list[float] | None
        Angles in degrees. Default: [30, 45, 60, 90, 120, 150].
    arm_length_mm : float
        Length of each corner arm.
    origin : tuple[float, float]
        Starting position.
    spacing_mm : float
        Horizontal spacing between corners.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations for corner test.
    """
    if angles is None:
        angles = [30.0, 45.0, 60.0, 90.0, 120.0, 150.0]

    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    for i, angle in enumerate(angles):
        cx = x0 + i * spacing_mm

        # Draw V shape centered at (cx, y0)
        half_angle_rad = math.radians(angle / 2)

        # Left arm
        lx = cx - arm_length_mm * math.sin(half_angle_rad)
        ly = y0 + arm_length_mm * math.cos(half_angle_rad)

        # Right arm
        rx = cx + arm_length_mm * math.sin(half_angle_rad)
        ry = y0 + arm_length_mm * math.cos(half_angle_rad)

        # Draw from left arm tip, through vertex, to right arm tip
        ops.extend(_stroke([(lx, ly), (cx, y0), (rx, ry)], feed))

    return ops


def fine_detail_test(
    spacing_mm: list[float] | None = None,
    line_length_mm: float = 30.0,
    num_lines: int = 5,
    origin: tuple[float, float] = (50.0, 100.0),
    group_spacing_mm: float = 20.0,
    feed: float | None = None,
) -> list[Operation]:
    """
    Draw groups of closely spaced parallel lines.

    Tests minimum feature size and line merge behavior.

    Parameters
    ----------
    spacing_mm : list[float] | None
        Line spacings to test. Default: [0.5, 1.0, 1.5, 2.0, 3.0].
    line_length_mm : float
        Length of each line.
    num_lines : int
        Number of lines per group.
    origin : tuple[float, float]
        Starting position.
    group_spacing_mm : float
        Horizontal spacing between groups.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations for fine detail test.
    """
    if spacing_mm is None:
        spacing_mm = [0.5, 1.0, 1.5, 2.0, 3.0]

    ops: list[Operation] = [ToolUp()]
    x0, y0 = origin

    group_x = x0
    for spacing in spacing_mm:
        # Draw vertical lines with this spacing
        for i in range(num_lines):
            x = group_x + i * spacing
            ops.extend(_stroke([(x, y0), (x, y0 + line_length_mm)], feed))

        group_x += (num_lines - 1) * spacing + group_spacing_mm

    return ops


# --- Convenience Functions ---


def full_calibration_suite(
    canvas_center: tuple[float, float] = (105.0, 148.5),
    feed: float | None = None,
) -> list[Operation]:
    """
    Generate a complete calibration pattern suite.

    Combines multiple patterns for comprehensive calibration.

    Parameters
    ----------
    canvas_center : tuple[float, float]
        Canvas center for pattern placement.
    feed : float | None
        Drawing feed rate in mm/min.

    Returns
    -------
    list[Operation]
        Job IR operations for full calibration suite.
    """
    ops: list[Operation] = [ToolUp()]

    # Center cross
    ops.extend(cross(size_mm=30, center=canvas_center, feed=feed))

    # Corner squares (25mm at each corner area)
    for ox, oy in [(20, 20), (165, 20), (20, 252), (165, 252)]:
        ops.extend(square(size_mm=25, origin=(ox, oy), feed=feed))

    # Center circle
    ops.extend(circle(diameter_mm=40, center=canvas_center, feed=feed))

    # Rulers along edges
    ops.extend(ruler_x(length_mm=150, origin=(30, 30), feed=feed))
    ops.extend(ruler_y(length_mm=200, origin=(30, 50), feed=feed))

    return ops
