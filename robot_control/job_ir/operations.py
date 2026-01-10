"""
Job IR Operations.

Defines all possible job operations as immutable dataclasses. This vocabulary
is the contract between vector data and G-code generation.

Design Principles:
    - Semantic, not mechanical: ToolDown not "G1 Z20"
    - Units are millimeters: All dimensions in mm
    - Canvas-relative coordinates: Origin at canvas corner, not machine home
    - Tool-agnostic where possible: Same DrawPolyline for pen and airbrush
    - Extensible: Add airbrush operations without modifying existing ones
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class Operation(ABC):
    """Base class for all job operations."""

    pass


# --- Setup Operations ---


@dataclass(frozen=True, slots=True)
class HomeXY(Operation):
    """
    Home X and Y axes using limit switches.

    This should be called at the start of every job to establish a
    known machine reference position.
    """

    pass


@dataclass(frozen=True, slots=True)
class SelectTool(Operation):
    """
    Choose the active tool.

    Parameters
    ----------
    tool : {"pen", "airbrush"}
        The tool to select. Affects Z work height and XY offset.
    """

    tool: Literal["pen", "airbrush"]


# --- Tool Operations ---


@dataclass(frozen=True, slots=True)
class ToolUp(Operation):
    """
    Raise tool to safe travel height.

    The actual Z position is determined by the machine configuration's
    travel_z value. This operation must complete before any rapid moves.
    """

    pass


@dataclass(frozen=True, slots=True)
class ToolDown(Operation):
    """
    Lower tool to work height.

    The actual Z position depends on the currently selected tool:
    - pen: pen_work_z (pen contacts paper)
    - airbrush: airbrush_work_z (spray height above canvas)
    """

    pass


# --- Motion Operations ---


@dataclass(frozen=True, slots=True)
class RapidXY(Operation):
    """
    Fast travel move to a position.

    The tool must be in the up position before executing this operation.
    Uses the tool's configured travel_feed rate.

    Parameters
    ----------
    x : float
        Target X position in mm, canvas-relative.
    y : float
        Target Y position in mm, canvas-relative.
    """

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LinearMove(Operation):
    """
    Single line segment at draw speed.

    Used for drawing when the polyline has only one segment,
    or for individual moves during calibration.

    Parameters
    ----------
    x : float
        Target X position in mm, canvas-relative.
    y : float
        Target Y position in mm, canvas-relative.
    feed : float | None
        Feed rate in mm/min. If None, uses the tool's default feed rate.
    """

    x: float
    y: float
    feed: float | None = None


@dataclass(frozen=True, slots=True)
class DrawPolyline(Operation):
    """
    Connected line segments forming a polyline.

    The tool should be down before executing this operation.
    Each point is connected to the next with a linear move.

    Parameters
    ----------
    points : tuple[tuple[float, float], ...]
        Sequence of (x, y) coordinates in mm, canvas-relative.
        Must have at least 2 points.
    feed : float | None
        Feed rate in mm/min. If None, uses the tool's default feed rate.
    """

    points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    feed: float | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError(
                f"DrawPolyline requires at least 2 points, got {len(self.points)}"
            )


# --- Future: Airbrush Operations (placeholders) ---


@dataclass(frozen=True, slots=True)
class SetInkMix(Operation):
    """
    Set CMY ratio for spray (future airbrush support).

    Parameters
    ----------
    c : float
        Cyan ratio, 0.0 to 1.0.
    m : float
        Magenta ratio, 0.0 to 1.0.
    y : float
        Yellow ratio, 0.0 to 1.0.
    """

    c: float
    m: float
    y: float

    def __post_init__(self) -> None:
        for name, val in [("c", self.c), ("m", self.m), ("y", self.y)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass(frozen=True, slots=True)
class SprayOn(Operation):
    """Begin spraying (future airbrush support)."""

    pass


@dataclass(frozen=True, slots=True)
class SprayOff(Operation):
    """Stop spraying (future airbrush support)."""

    pass


# --- Type Aliases ---

# A stroke is a logical drawing unit: travel to start, lower tool, draw, raise tool
Stroke = list[Operation]

# A job is a sequence of strokes
Job = list[Stroke]


def create_stroke(
    start_x: float,
    start_y: float,
    points: list[tuple[float, float]],
    feed: float | None = None,
) -> Stroke:
    """
    Create a complete stroke from a starting position and polyline points.

    A stroke consists of:
    1. Rapid move to start position (tool must already be up)
    2. Lower tool
    3. Draw the polyline
    4. Raise tool

    Parameters
    ----------
    start_x : float
        Starting X position in mm, canvas-relative.
    start_y : float
        Starting Y position in mm, canvas-relative.
    points : list[tuple[float, float]]
        Polyline points including the start. Must have >= 2 points.
    feed : float | None
        Feed rate in mm/min for drawing. None uses tool default.

    Returns
    -------
    Stroke
        List of operations forming a complete stroke.
    """
    if len(points) < 2:
        raise ValueError(f"Stroke requires at least 2 points, got {len(points)}")

    return [
        RapidXY(x=start_x, y=start_y),
        ToolDown(),
        DrawPolyline(points=tuple(points), feed=feed),
        ToolUp(),
    ]


def operations_to_strokes(operations: list[Operation]) -> Job:
    """
    Group a flat list of operations into strokes.

    Strokes are delimited by ToolUp operations. Each stroke starts after
    a ToolUp and ends with the next ToolUp (inclusive).

    Parameters
    ----------
    operations : list[Operation]
        Flat list of operations.

    Returns
    -------
    Job
        List of strokes (lists of operations).
    """
    if not operations:
        return []

    strokes: Job = []
    current_stroke: Stroke = []

    for op in operations:
        current_stroke.append(op)
        if isinstance(op, ToolUp):
            strokes.append(current_stroke)
            current_stroke = []

    # Handle any trailing operations without a final ToolUp
    if current_stroke:
        strokes.append(current_stroke)

    return strokes
