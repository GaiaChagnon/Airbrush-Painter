"""Tests for calibration patterns and measurement helpers.

Validates pattern structure, measurement calculations, and that all
patterns produce valid Job IR operations.
"""

from __future__ import annotations

import math

import pytest

from robot_control.calibration import patterns
from robot_control.calibration.measurement import (
    MeasurementResult,
    PlanarityResult,
    calculate_new_rotation_distance,
    calculate_steps_correction,
    calculate_tool_offset,
    check_planarity,
    format_calibration_summary,
)
from robot_control.job_ir.operations import (
    DrawPolyline,
    LinearMove,
    Operation,
    RapidXY,
    ToolDown,
    ToolUp,
)


# ---------------------------------------------------------------------------
# Pattern structure tests
# ---------------------------------------------------------------------------


class TestBasicPatterns:
    def test_square_structure(self) -> None:
        ops = patterns.square(size_mm=50.0, origin=(0.0, 0.0))
        assert isinstance(ops[0], ToolUp)
        has_rapid = any(isinstance(op, RapidXY) for op in ops)
        has_down = any(isinstance(op, ToolDown) for op in ops)
        has_poly = any(isinstance(op, DrawPolyline) for op in ops)
        assert has_rapid and has_down and has_poly

    def test_square_dimensions(self) -> None:
        size = 50.0
        origin = (10.0, 20.0)
        ops = patterns.square(size_mm=size, origin=origin)
        poly = next(op for op in ops if isinstance(op, DrawPolyline))
        xs = [p[0] for p in poly.points]
        ys = [p[1] for p in poly.points]
        assert pytest.approx(max(xs) - min(xs)) == size
        assert pytest.approx(max(ys) - min(ys)) == size

    def test_rectangle(self) -> None:
        ops = patterns.rectangle(width=60.0, height=40.0)
        poly = next(op for op in ops if isinstance(op, DrawPolyline))
        xs = [p[0] for p in poly.points]
        ys = [p[1] for p in poly.points]
        assert pytest.approx(max(xs) - min(xs)) == 60.0
        assert pytest.approx(max(ys) - min(ys)) == 40.0

    def test_cross(self) -> None:
        ops = patterns.cross(size_mm=30.0, origin=(50.0, 50.0))
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 2  # horizontal + vertical

    def test_grid(self) -> None:
        ops = patterns.grid(rows=3, cols=3, spacing=10.0)
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 6  # 3 horizontal + 3 vertical

    def test_diagonal(self) -> None:
        ops = patterns.diagonal(size_mm=100.0)
        poly = next(op for op in ops if isinstance(op, DrawPolyline))
        assert len(poly.points) == 2

    def test_circle_closure(self) -> None:
        ops = patterns.circle(diameter=50.0, segments=36)
        poly = next(op for op in ops if isinstance(op, DrawPolyline))
        # First and last points should be the same (closed)
        assert pytest.approx(poly.points[0][0], abs=0.01) == poly.points[-1][0]
        assert pytest.approx(poly.points[0][1], abs=0.01) == poly.points[-1][1]


class TestCalibrationPatterns:
    def test_ruler_x_has_ticks(self) -> None:
        ops = patterns.ruler_x(length=100.0, tick_spacing=10.0)
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        # Main line + 11 ticks (0, 10, 20, ..., 100)
        assert len(polys) >= 11

    def test_ruler_y(self) -> None:
        ops = patterns.ruler_y(length=100.0)
        assert any(isinstance(op, DrawPolyline) for op in ops)

    def test_crosshair_grid(self) -> None:
        ops = patterns.crosshair_grid(rows=2, cols=2)
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        # 4 crosshairs * 2 arms each = 8
        assert len(polys) == 8

    def test_speed_test(self) -> None:
        ops = patterns.speed_test(
            lengths=[50.0], speeds=[10.0, 20.0],
        )
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 2

    def test_backlash_test(self) -> None:
        ops = patterns.backlash_test(repeats=3)
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 6  # 3 forward + 3 reverse

    def test_z_touch_default_positions(self) -> None:
        ops = patterns.z_touch_pattern()
        # Should have 5 touch points (centre + 4 corners)
        tool_downs = [op for op in ops if isinstance(op, ToolDown)]
        assert len(tool_downs) == 5


class TestPenPatterns:
    def test_line_weight(self) -> None:
        ops = patterns.line_weight_test(speeds=[10.0, 20.0, 30.0])
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 3

    def test_corner_test(self) -> None:
        ops = patterns.corner_test(angles=[45.0, 90.0])
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 2

    def test_fine_detail(self) -> None:
        ops = patterns.fine_detail_test(spacings=[1.0, 0.5], num_lines=3)
        polys = [op for op in ops if isinstance(op, DrawPolyline)]
        assert len(polys) == 6  # 2 groups * 3 lines


class TestCalibrationSuite:
    def test_suite_runs(self) -> None:
        ops = patterns.calibration_suite()
        assert len(ops) > 10
        assert any(isinstance(op, DrawPolyline) for op in ops)


# ---------------------------------------------------------------------------
# Measurement calculation tests
# ---------------------------------------------------------------------------


class TestMeasurementCalcs:
    def test_steps_correction_perfect(self) -> None:
        result = calculate_steps_correction(100.0, 100.0)
        assert result.error == 0.0
        assert result.correction_factor == 1.0

    def test_steps_correction_short(self) -> None:
        result = calculate_steps_correction(100.0, 99.0)
        assert result.error == pytest.approx(1.0)
        assert result.correction_factor == pytest.approx(100.0 / 99.0)

    def test_steps_correction_long(self) -> None:
        result = calculate_steps_correction(100.0, 101.0)
        assert result.error == pytest.approx(-1.0)
        assert result.correction_factor == pytest.approx(100.0 / 101.0)

    def test_steps_correction_zero_measured(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            calculate_steps_correction(100.0, 0.0)

    def test_new_rotation_distance(self) -> None:
        new_rd = calculate_new_rotation_distance(32.0, 100.0, 99.5)
        expected = 32.0 * (100.0 / 99.5)
        assert new_rd == pytest.approx(expected)

    def test_new_rotation_distance_zero_measured(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            calculate_new_rotation_distance(32.0, 100.0, 0.0)

    def test_tool_offset(self) -> None:
        ox, oy = calculate_tool_offset(10.0, 20.0, 10.5, 19.5)
        assert ox == pytest.approx(-0.5)
        assert oy == pytest.approx(0.5)

    def test_format_summary(self) -> None:
        text = format_calibration_summary({"key": 1.234, "name": "test"})
        assert "key" in text
        assert "1.234" in text
        assert "test" in text


# ---------------------------------------------------------------------------
# Planarity check tests
# ---------------------------------------------------------------------------


class TestPlanarityCheck:
    def test_perfectly_flat(self) -> None:
        """All points at the same Z -> zero residuals, pass."""
        pts = [
            (0.0, 0.0, 80.0),
            (100.0, 0.0, 80.0),
            (200.0, 0.0, 80.0),
            (0.0, 100.0, 80.0),
            (100.0, 100.0, 80.0),
            (200.0, 100.0, 80.0),
        ]
        result = check_planarity(pts, tolerance=0.01)
        assert result.is_planar
        assert result.max_residual_mm < 1e-10

    def test_tilted_plane(self) -> None:
        """Points on a tilted plane -> zero residuals, pass."""
        pts = [
            (0.0, 0.0, 80.0),
            (450.0, 0.0, 80.1),
            (0.0, 300.0, 79.9),
            (450.0, 300.0, 80.0),
        ]
        result = check_planarity(pts, tolerance=0.01)
        assert result.is_planar
        assert result.max_residual_mm < 1e-6

    def test_warped_surface(self) -> None:
        """One point off-plane -> residual exceeds tolerance, warning."""
        pts = [
            (0.0, 0.0, 80.0),
            (225.0, 0.0, 80.0),
            (450.0, 0.0, 80.0),
            (0.0, 150.0, 80.0),
            (225.0, 150.0, 80.5),  # warped
            (450.0, 150.0, 80.0),
            (0.0, 300.0, 80.0),
            (225.0, 300.0, 80.0),
            (450.0, 300.0, 80.0),
        ]
        result = check_planarity(pts, tolerance=0.15)
        assert not result.is_planar
        assert result.max_residual_mm > 0.15

    def test_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="Need >= 3"):
            check_planarity([(0, 0, 1), (1, 0, 1)])

    def test_returns_plane_coefficients(self) -> None:
        """Plane z = 0.001*x + 80."""
        pts = [
            (0.0, 0.0, 80.0),
            (100.0, 0.0, 80.1),
            (200.0, 0.0, 80.2),
            (0.0, 100.0, 80.0),
            (200.0, 100.0, 80.2),
        ]
        result = check_planarity(pts)
        a, b, c = result.plane_coeffs
        assert pytest.approx(a, abs=1e-6) == 0.001
        assert pytest.approx(b, abs=1e-6) == 0.0
        assert pytest.approx(c, abs=1e-3) == 80.0
