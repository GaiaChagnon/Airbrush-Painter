"""Tests for Job IR operations module.

Validates dataclass creation, immutability, validation, and stroke helpers.
"""

from __future__ import annotations

import pytest

from robot_control.job_ir.operations import (
    DrawArc,
    DrawPolyline,
    HomeXY,
    LinearMove,
    Operation,
    Purge,
    RapidXY,
    RefillPump,
    SelectTool,
    SetInkMix,
    SprayOff,
    SprayOn,
    ToolDown,
    ToolUp,
    create_stroke,
    operations_to_strokes,
)


# ---------------------------------------------------------------------------
# Dataclass creation and immutability
# ---------------------------------------------------------------------------


class TestOperationDataclasses:
    def test_home_xy(self) -> None:
        op = HomeXY()
        assert isinstance(op, Operation)

    def test_select_tool_valid(self) -> None:
        pen = SelectTool(tool="pen")
        assert pen.tool == "pen"
        airbrush = SelectTool(tool="airbrush")
        assert airbrush.tool == "airbrush"

    def test_select_tool_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be 'pen' or 'airbrush'"):
            SelectTool(tool="laser")

    def test_tool_up_down(self) -> None:
        up = ToolUp()
        down = ToolDown()
        assert isinstance(up, Operation)
        assert isinstance(down, Operation)

    def test_rapid_xy(self) -> None:
        op = RapidXY(x=10.5, y=20.3)
        assert op.x == 10.5
        assert op.y == 20.3

    def test_linear_move_with_feed(self) -> None:
        op = LinearMove(x=5.0, y=6.0, feed=30.0)
        assert op.feed == 30.0

    def test_linear_move_default_feed(self) -> None:
        op = LinearMove(x=5.0, y=6.0)
        assert op.feed is None

    def test_draw_polyline(self) -> None:
        pts = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
        op = DrawPolyline(points=pts)
        assert len(op.points) == 3

    def test_draw_polyline_too_few_points(self) -> None:
        with pytest.raises(ValueError, match=">=\\s*2 points"):
            DrawPolyline(points=((0.0, 0.0),))

    def test_draw_arc_clockwise(self) -> None:
        op = DrawArc(x=10.0, y=0.0, i=5.0, j=0.0, clockwise=True)
        assert op.clockwise is True
        assert op.x == 10.0
        assert op.i == 5.0
        assert op.feed is None

    def test_draw_arc_ccw_with_feed(self) -> None:
        op = DrawArc(x=0.0, y=10.0, i=0.0, j=5.0, clockwise=False, feed=30.0)
        assert op.clockwise is False
        assert op.feed == 30.0

    def test_draw_arc_frozen(self) -> None:
        op = DrawArc(x=1.0, y=2.0, i=3.0, j=4.0)
        with pytest.raises(AttributeError):
            op.x = 99.0  # type: ignore[misc]

    def test_frozen(self) -> None:
        op = RapidXY(x=1.0, y=2.0)
        with pytest.raises(AttributeError):
            op.x = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Airbrush / pump operations
# ---------------------------------------------------------------------------


class TestAirbrushOps:
    def test_set_ink_mix(self) -> None:
        op = SetInkMix(c=0.5, m=0.3, y=0.2)
        assert op.c == 0.5

    def test_set_ink_mix_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            SetInkMix(c=1.5, m=0.0, y=0.0)

    def test_spray_on_off(self) -> None:
        assert isinstance(SprayOn(), Operation)
        assert isinstance(SprayOff(), Operation)

    def test_purge(self) -> None:
        op = Purge(volume_ml=5.0)
        assert op.volume_ml == 5.0

    def test_refill_prime(self) -> None:
        assert RefillPump(pump="cyan").pump == "cyan"


# ---------------------------------------------------------------------------
# Stroke helpers
# ---------------------------------------------------------------------------


class TestCreateStroke:
    def test_basic_stroke(self) -> None:
        stroke = create_stroke([(0, 0), (10, 0), (10, 10)])
        assert isinstance(stroke[0], RapidXY)
        assert isinstance(stroke[1], ToolDown)
        assert isinstance(stroke[2], DrawPolyline)
        assert isinstance(stroke[3], ToolUp)

    def test_stroke_with_feed(self) -> None:
        stroke = create_stroke([(0, 0), (10, 0)], feed=42.0)
        poly = [op for op in stroke if isinstance(op, DrawPolyline)][0]
        assert poly.feed == 42.0

    def test_stroke_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            create_stroke([(0, 0)])


class TestOperationsToStrokes:
    def test_split_on_tool_up(self) -> None:
        ops = [
            RapidXY(x=0, y=0),
            ToolDown(),
            DrawPolyline(points=((0, 0), (10, 0))),
            ToolUp(),
            RapidXY(x=20, y=0),
            ToolDown(),
            DrawPolyline(points=((20, 0), (30, 0))),
            ToolUp(),
        ]
        strokes = operations_to_strokes(ops)
        assert len(strokes) == 2

    def test_empty_list(self) -> None:
        assert operations_to_strokes([]) == []

    def test_trailing_ops_without_tool_up(self) -> None:
        ops = [
            RapidXY(x=0, y=0),
            ToolDown(),
            DrawPolyline(points=((0, 0), (10, 0))),
        ]
        strokes = operations_to_strokes(ops)
        assert len(strokes) == 1

    def test_setup_ops_in_own_group(self) -> None:
        ops = [
            HomeXY(),
            SelectTool(tool="pen"),
            ToolUp(),
            RapidXY(x=0, y=0),
            ToolDown(),
            DrawPolyline(points=((0, 0), (10, 0))),
            ToolUp(),
        ]
        strokes = operations_to_strokes(ops)
        # First group: HomeXY, SelectTool, ToolUp
        assert isinstance(strokes[0][-1], ToolUp)
        assert len(strokes) == 2
