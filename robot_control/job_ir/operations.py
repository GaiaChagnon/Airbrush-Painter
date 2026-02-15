"""Job IR operations -- the vocabulary between vector data and G-code.

Every possible job action is an immutable, slotted dataclass.  Operations
use **semantic** names (``ToolDown``, not ``G1 Z20``), **millimetre**
units, and **canvas-relative** coordinates (origin at canvas corner, not
machine home).

Grouping
--------
A *Stroke* is a list of operations that form one atomic drawing unit
(rapid to start, tool down, draw, tool up).  The executor processes one
stroke at a time in interactive mode.

Extension
---------
Airbrush operations (``SetInkMix``, ``SprayOn``, etc.) are defined but
unused in pen-only mode.  Adding them later does not modify existing ops.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Stroke = list["Operation"]
"""One atomic drawing unit processed by the interactive executor."""

Job = list[Stroke]
"""A complete job is a sequence of strokes."""

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Operation(ABC):
    """Base class for all job operations."""

    pass


# ---------------------------------------------------------------------------
# Setup operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HomeXY(Operation):
    """Home X and Y axes using limit switches.

    Must be called at the start of every job to establish a known
    machine reference position.  Does **not** home Z (no endstop yet).
    """

    pass


@dataclass(frozen=True, slots=True)
class SelectTool(Operation):
    """Choose the active tool.

    Parameters
    ----------
    tool : ``"pen"`` | ``"airbrush"``
        Tool to activate.  Affects subsequent Z-state mapping and
        XY-offset application in the G-code generator.
    """

    tool: Literal["pen", "airbrush"]

    def __post_init__(self) -> None:
        if self.tool not in ("pen", "airbrush"):
            raise ValueError(
                f"tool must be 'pen' or 'airbrush', got {self.tool!r}"
            )


# ---------------------------------------------------------------------------
# Motion operations  (all coordinates are canvas-relative mm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RapidXY(Operation):
    """Fast travel move -- tool **must** be up.

    Parameters
    ----------
    x, y : float
        Target position in canvas mm (top-left origin, +Y down).
    """

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LinearMove(Operation):
    """Single line segment at draw speed.

    Parameters
    ----------
    x, y : float
        End-point in canvas mm.
    feed : float | None
        Override feed rate (mm/s).  ``None`` uses the active tool default.
    """

    x: float
    y: float
    feed: float | None = None


@dataclass(frozen=True, slots=True)
class DrawPolyline(Operation):
    """Connected line segments (pen-down drawing).

    Parameters
    ----------
    points : tuple[tuple[float, float], ...]
        Ordered vertices in canvas mm.  Must contain >= 2 points.
    feed : float | None
        Override feed rate (mm/s).  ``None`` uses the active tool default.
    """

    points: tuple[tuple[float, float], ...]
    feed: float | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError(
                f"DrawPolyline requires >= 2 points, got {len(self.points)}"
            )


@dataclass(frozen=True, slots=True)
class DrawArc(Operation):
    """Circular arc at draw speed (G2/G3).

    Uses the centre-offset (I/J) form.  The arc is drawn from the
    current position to ``(x, y)`` with the centre of curvature at
    ``(current + i, current + j)``.

    Parameters
    ----------
    x, y : float
        End-point in canvas mm.
    i, j : float
        Offset from the current position to the arc centre, in mm.
        Signs matter: positive I is +X, positive J is +Y.
    clockwise : bool
        ``True`` for G2 (clockwise), ``False`` for G3 (counter-clockwise).
    feed : float | None
        Override feed rate (mm/s).  ``None`` uses the active tool default.
    """

    x: float
    y: float
    i: float
    j: float
    clockwise: bool = True
    feed: float | None = None


# ---------------------------------------------------------------------------
# Tool operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolUp(Operation):
    """Raise tool to safe (travel) height."""

    pass


@dataclass(frozen=True, slots=True)
class ToolDown(Operation):
    """Lower tool to work height (pen contacts paper / airbrush spray)."""

    pass


# ---------------------------------------------------------------------------
# Future: airbrush operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SetInkMix(Operation):
    """Set CMY ratio for spray.

    Parameters
    ----------
    c, m, y : float
        Cyan, magenta, yellow fractions in [0, 1].
    """

    c: float
    m: float
    y: float

    def __post_init__(self) -> None:
        for ch, val in [("c", self.c), ("m", self.m), ("y", self.y)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"SetInkMix {ch} must be in [0, 1], got {val}"
                )


@dataclass(frozen=True, slots=True)
class SprayOn(Operation):
    """Begin spraying (airbrush valve open)."""

    pass


@dataclass(frozen=True, slots=True)
class SprayOff(Operation):
    """Stop spraying (airbrush valve closed)."""

    pass


@dataclass(frozen=True, slots=True)
class Purge(Operation):
    """Flush with solvent.

    Parameters
    ----------
    volume_ml : float
        Solvent volume to push through.
    """

    volume_ml: float


# ---------------------------------------------------------------------------
# Future: pump operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefillPump(Operation):
    """Execute refill sequence for a pump.

    Parameters
    ----------
    pump : str
        Pump identifier (e.g., ``"cyan"``).
    """

    pump: str


@dataclass(frozen=True, slots=True)
class PrimeLine(Operation):
    """Push ink to the nozzle.

    Parameters
    ----------
    pump : str
        Pump identifier.
    """

    pump: str


# ---------------------------------------------------------------------------
# Stroke helpers
# ---------------------------------------------------------------------------


def create_stroke(
    points: list[tuple[float, float]],
    feed: float | None = None,
) -> Stroke:
    """Build a standard stroke: rapid -> tool-down -> polyline -> tool-up.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Ordered polyline vertices (canvas mm).  Must have >= 2 points.
    feed : float | None
        Override drawing feed rate (mm/s).

    Returns
    -------
    Stroke
        ``[RapidXY, ToolDown, DrawPolyline, ToolUp]``
    """
    if len(points) < 2:
        raise ValueError("Stroke requires at least 2 points")

    return [
        RapidXY(x=points[0][0], y=points[0][1]),
        ToolDown(),
        DrawPolyline(points=tuple(points), feed=feed),
        ToolUp(),
    ]


def operations_to_strokes(ops: list[Operation]) -> Job:
    """Split a flat operation list into strokes at ``ToolUp`` boundaries.

    A stroke is delimited by the sequence of ops between (and including)
    each ``ToolUp``.  Leading setup ops (``HomeXY``, ``SelectTool``)
    before the first drawing stroke are placed in their own group.

    Parameters
    ----------
    ops : list[Operation]
        Flat operation list.

    Returns
    -------
    Job
        List of strokes (each a ``list[Operation]``).
    """
    if not ops:
        return []

    strokes: Job = []
    current: Stroke = []

    for op in ops:
        current.append(op)
        if isinstance(op, ToolUp):
            strokes.append(current)
            current = []

    # Residual ops that didn't end with ToolUp
    if current:
        strokes.append(current)

    return strokes
