"""
Tests for Job IR operations module.
"""

import pytest

from robot_control.job_ir.operations import (
    DrawPolyline,
    HomeXY,
    LinearMove,
    Operation,
    RapidXY,
    SelectTool,
    SetInkMix,
    ToolDown,
    ToolUp,
    create_stroke,
    operations_to_strokes,
)


class TestOperationDataclasses:
    """Test operation dataclass definitions."""

    def test_home_xy_creation(self):
        """HomeXY should be creatable with no args."""
        op = HomeXY()
        assert isinstance(op, Operation)

    def test_select_tool_valid(self):
        """SelectTool should accept 'pen' or 'airbrush'."""
        pen = SelectTool(tool="pen")
        airbrush = SelectTool(tool="airbrush")
        assert pen.tool == "pen"
        assert airbrush.tool == "airbrush"

    def test_tool_up_down(self):
        """ToolUp and ToolDown should be creatable."""
        up = ToolUp()
        down = ToolDown()
        assert isinstance(up, Operation)
        assert isinstance(down, Operation)

    def test_rapid_xy(self):
        """RapidXY should store coordinates."""
        op = RapidXY(x=10.5, y=20.3)
        assert op.x == 10.5
        assert op.y == 20.3

    def test_linear_move_with_feed(self):
        """LinearMove should accept optional feed rate."""
        op1 = LinearMove(x=50.0, y=75.0)
        op2 = LinearMove(x=50.0, y=75.0, feed=1500.0)

        assert op1.feed is None
        assert op2.feed == 1500.0

    def test_draw_polyline_valid(self):
        """DrawPolyline should accept 2+ points."""
        points = ((0.0, 0.0), (10.0, 10.0), (20.0, 0.0))
        op = DrawPolyline(points=points)
        assert len(op.points) == 3
        assert op.points[0] == (0.0, 0.0)

    def test_draw_polyline_requires_two_points(self):
        """DrawPolyline should reject less than 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            DrawPolyline(points=((0.0, 0.0),))

    def test_draw_polyline_empty_rejected(self):
        """DrawPolyline should reject empty points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            DrawPolyline(points=())

    def test_operations_are_frozen(self):
        """Operations should be immutable."""
        op = RapidXY(x=10.0, y=20.0)
        with pytest.raises(AttributeError):
            op.x = 50.0  # type: ignore


class TestSetInkMix:
    """Test SetInkMix validation."""

    def test_valid_ink_mix(self):
        """SetInkMix should accept values in [0, 1]."""
        mix = SetInkMix(c=0.5, m=0.3, y=0.8)
        assert mix.c == 0.5
        assert mix.m == 0.3
        assert mix.y == 0.8

    def test_ink_mix_boundary_values(self):
        """SetInkMix should accept 0.0 and 1.0."""
        mix = SetInkMix(c=0.0, m=1.0, y=0.0)
        assert mix.c == 0.0
        assert mix.m == 1.0

    def test_ink_mix_rejects_negative(self):
        """SetInkMix should reject negative values."""
        with pytest.raises(ValueError, match="must be in"):
            SetInkMix(c=-0.1, m=0.5, y=0.5)

    def test_ink_mix_rejects_over_one(self):
        """SetInkMix should reject values over 1."""
        with pytest.raises(ValueError, match="must be in"):
            SetInkMix(c=0.5, m=1.5, y=0.5)


class TestCreateStroke:
    """Test stroke creation helper."""

    def test_create_stroke_basic(self):
        """create_stroke should generate proper operation sequence."""
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        stroke = create_stroke(start_x=0.0, start_y=0.0, points=points)

        assert len(stroke) == 4
        assert isinstance(stroke[0], RapidXY)
        assert isinstance(stroke[1], ToolDown)
        assert isinstance(stroke[2], DrawPolyline)
        assert isinstance(stroke[3], ToolUp)

    def test_create_stroke_with_feed(self):
        """create_stroke should pass feed to DrawPolyline."""
        points = [(0.0, 0.0), (10.0, 10.0)]
        stroke = create_stroke(start_x=0.0, start_y=0.0, points=points, feed=2000.0)

        polyline = stroke[2]
        assert isinstance(polyline, DrawPolyline)
        assert polyline.feed == 2000.0

    def test_create_stroke_requires_two_points(self):
        """create_stroke should require at least 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            create_stroke(start_x=0.0, start_y=0.0, points=[(0.0, 0.0)])


class TestOperationsToStrokes:
    """Test stroke grouping."""

    def test_single_stroke(self):
        """Single stroke should be grouped correctly."""
        ops = [
            RapidXY(x=0, y=0),
            ToolDown(),
            LinearMove(x=10, y=10),
            ToolUp(),
        ]

        strokes = operations_to_strokes(ops)
        assert len(strokes) == 1
        assert len(strokes[0]) == 4

    def test_multiple_strokes(self):
        """Multiple strokes should be separated by ToolUp."""
        ops = [
            RapidXY(x=0, y=0),
            ToolDown(),
            LinearMove(x=10, y=10),
            ToolUp(),
            RapidXY(x=20, y=20),
            ToolDown(),
            LinearMove(x=30, y=30),
            ToolUp(),
        ]

        strokes = operations_to_strokes(ops)
        assert len(strokes) == 2
        assert len(strokes[0]) == 4
        assert len(strokes[1]) == 4

    def test_trailing_operations(self):
        """Operations after last ToolUp should form final stroke."""
        ops = [
            RapidXY(x=0, y=0),
            ToolDown(),
        ]

        strokes = operations_to_strokes(ops)
        assert len(strokes) == 1
        assert len(strokes[0]) == 2

    def test_empty_operations(self):
        """Empty operations should return empty list."""
        strokes = operations_to_strokes([])
        assert strokes == []
