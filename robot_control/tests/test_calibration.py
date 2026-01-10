"""
Tests for calibration patterns module.
"""

import math

import pytest

from robot_control.calibration import patterns
from robot_control.job_ir.operations import (
    DrawPolyline,
    LinearMove,
    Operation,
    RapidXY,
    ToolDown,
    ToolUp,
)


class TestBasicPatterns:
    """Test basic geometry patterns."""

    def test_square_structure(self):
        """Square should generate correct operation sequence."""
        ops = patterns.square(size_mm=50.0, origin=(0.0, 0.0))

        # Should start with ToolUp
        assert isinstance(ops[0], ToolUp)

        # Should contain RapidXY, ToolDown, DrawPolyline, ToolUp
        has_rapid = any(isinstance(op, RapidXY) for op in ops)
        has_down = any(isinstance(op, ToolDown) for op in ops)
        has_polyline = any(isinstance(op, DrawPolyline) for op in ops)
        has_up = sum(1 for op in ops if isinstance(op, ToolUp))

        assert has_rapid
        assert has_down
        assert has_polyline
        assert has_up >= 2  # Initial + after stroke

    def test_square_dimensions(self):
        """Square should have correct dimensions."""
        size = 50.0
        origin = (10.0, 20.0)
        ops = patterns.square(size_mm=size, origin=origin)

        # Find the polyline
        polyline = next(op for op in ops if isinstance(op, DrawPolyline))

        # Check corners
        points = polyline.points
        assert len(points) == 5  # Closed square

        # Verify dimensions
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        assert min(x_coords) == origin[0]
        assert max(x_coords) == origin[0] + size
        assert min(y_coords) == origin[1]
        assert max(y_coords) == origin[1] + size

    def test_rectangle(self):
        """Rectangle should have correct dimensions."""
        width, height = 80.0, 60.0
        origin = (0.0, 0.0)
        ops = patterns.rectangle(width_mm=width, height_mm=height, origin=origin)

        polyline = next(op for op in ops if isinstance(op, DrawPolyline))
        points = polyline.points

        x_range = max(p[0] for p in points) - min(p[0] for p in points)
        y_range = max(p[1] for p in points) - min(p[1] for p in points)

        assert abs(x_range - width) < 0.01
        assert abs(y_range - height) < 0.01

    def test_cross(self):
        """Cross should have two perpendicular lines."""
        size = 50.0
        center = (100.0, 100.0)
        ops = patterns.cross(size_mm=size, center=center)

        # Should have 2 strokes (2 lines)
        polylines = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polylines) == 2

        # Each line should be size_mm long
        for pl in polylines:
            p1, p2 = pl.points[0], pl.points[-1]
            length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            assert abs(length - size) < 0.01

    def test_grid(self):
        """Grid should have correct number of lines."""
        rows, cols = 3, 4
        ops = patterns.grid(rows=rows, cols=cols, spacing_mm=10.0)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]

        # Should have rows horizontal + cols vertical lines
        assert len(polylines) == rows + cols

    def test_circle(self):
        """Circle should approximate a closed curve."""
        diameter = 50.0
        segments = 32
        center = (100.0, 100.0)
        ops = patterns.circle(
            diameter_mm=diameter, center=center, segments=segments
        )

        polyline = next(op for op in ops if isinstance(op, DrawPolyline))

        # Should have segments + 1 points (closed)
        assert len(polyline.points) == segments + 1

        # First and last points should be the same
        p_first, p_last = polyline.points[0], polyline.points[-1]
        assert abs(p_first[0] - p_last[0]) < 0.01
        assert abs(p_first[1] - p_last[1]) < 0.01

        # All points should be radius distance from center
        radius = diameter / 2
        for x, y in polyline.points:
            dist = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            assert abs(dist - radius) < 0.01


class TestCalibrationPatterns:
    """Test calibration-specific patterns."""

    def test_ruler_x(self):
        """X ruler should have horizontal line and ticks."""
        length = 100.0
        tick_spacing = 10.0
        ops = patterns.ruler_x(
            length_mm=length,
            tick_spacing_mm=tick_spacing,
            origin=(0.0, 0.0),
        )

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]

        # Should have main line + ticks
        num_ticks = int(length / tick_spacing) + 1
        assert len(polylines) == 1 + num_ticks

    def test_ruler_y(self):
        """Y ruler should have vertical line and ticks."""
        length = 100.0
        tick_spacing = 10.0
        ops = patterns.ruler_y(
            length_mm=length,
            tick_spacing_mm=tick_spacing,
            origin=(0.0, 0.0),
        )

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]
        num_ticks = int(length / tick_spacing) + 1
        assert len(polylines) == 1 + num_ticks

    def test_crosshair_grid(self):
        """Crosshair grid should have correct number of crosshairs."""
        rows, cols = 2, 3
        ops = patterns.crosshair_grid(rows=rows, cols=cols)

        # Each crosshair has 2 lines
        polylines = [op for op in ops if isinstance(op, DrawPolyline)]
        expected_lines = rows * cols * 2
        assert len(polylines) == expected_lines

    def test_speed_test(self):
        """Speed test should draw lines at different speeds."""
        lengths = [50.0, 100.0]
        speeds = [500.0, 1000.0, 1500.0]
        ops = patterns.speed_test(lengths_mm=lengths, speeds_mm_min=speeds)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]

        # Should have len(lengths) * len(speeds) lines
        expected = len(lengths) * len(speeds)
        assert len(polylines) == expected

        # Check that different feed rates are used
        feeds = set(pl.feed for pl in polylines if pl.feed is not None)
        assert feeds == set(speeds)

    def test_backlash_test(self):
        """Backlash test should create forward/reverse patterns."""
        distance = 20.0
        repeats = 3
        ops = patterns.backlash_test(distance_mm=distance, repeats=repeats)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]

        # Should have 2 patterns (horizontal and vertical)
        assert len(polylines) == 2


class TestPenPatterns:
    """Test pen-specific patterns."""

    def test_line_weight_test(self):
        """Line weight test should draw lines at different speeds."""
        speeds = [300.0, 500.0, 800.0]
        ops = patterns.line_weight_test(speeds_mm_min=speeds)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polylines) == len(speeds)

    def test_corner_test(self):
        """Corner test should draw corners at different angles."""
        angles = [30.0, 60.0, 90.0]
        ops = patterns.corner_test(angles=angles)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polylines) == len(angles)

    def test_fine_detail_test(self):
        """Fine detail test should draw groups of closely spaced lines."""
        spacings = [0.5, 1.0, 2.0]
        num_lines = 5
        ops = patterns.fine_detail_test(spacing_mm=spacings, num_lines=num_lines)

        polylines = [op for op in ops if isinstance(op, DrawPolyline)]

        # Each spacing group has num_lines lines
        expected = len(spacings) * num_lines
        assert len(polylines) == expected


class TestCalibrationSuite:
    """Test full calibration suite."""

    def test_full_suite_generates_operations(self):
        """Full calibration suite should generate valid operations."""
        ops = patterns.full_calibration_suite()

        assert len(ops) > 0

        # Should have various operation types
        op_types = set(type(op).__name__ for op in ops)
        assert "ToolUp" in op_types
        assert "DrawPolyline" in op_types

    def test_all_patterns_return_operations(self):
        """All pattern functions should return valid operation lists."""
        pattern_funcs = [
            patterns.square,
            patterns.rectangle,
            patterns.cross,
            patterns.grid,
            patterns.diagonal,
            patterns.circle,
            patterns.ruler_x,
            patterns.ruler_y,
            patterns.crosshair_grid,
            patterns.speed_test,
            patterns.backlash_test,
            patterns.z_touch_pattern,
            patterns.line_weight_test,
            patterns.corner_test,
            patterns.fine_detail_test,
            patterns.full_calibration_suite,
        ]

        for func in pattern_funcs:
            ops = func()
            assert isinstance(ops, list)
            assert len(ops) > 0
            assert all(isinstance(op, Operation) for op in ops)
