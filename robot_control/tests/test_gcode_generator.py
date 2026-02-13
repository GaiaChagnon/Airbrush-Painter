"""Tests for G-code generator.

Validates coordinate transforms, feed-rate conversion (mm/s -> mm/min),
soft-limit rejection, and direction-reversal pause insertion.
"""

from __future__ import annotations

import pytest

from robot_control.configs.loader import MachineConfig, load_config
from robot_control.gcode.generator import GCodeError, GCodeGenerator
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> MachineConfig:
    """Load the default machine config shipped with the package."""
    return load_config()


@pytest.fixture()
def gen(config: MachineConfig) -> GCodeGenerator:
    return GCodeGenerator(config)


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------


class TestHeaderFooter:
    def test_header_contains_g21_g90(self, gen: GCodeGenerator) -> None:
        gcode = gen.generate([ToolUp()])
        assert "G21" in gcode
        assert "G90" in gcode

    def test_footer_contains_m400(self, gen: GCodeGenerator) -> None:
        gcode = gen.generate([ToolUp()])
        lines = gcode.strip().splitlines()
        assert any("M400" in l for l in lines[-5:])


# ---------------------------------------------------------------------------
# Feed-rate conversion
# ---------------------------------------------------------------------------


class TestFeedConversion:
    def test_rapid_feed_in_mm_min(
        self, gen: GCodeGenerator, config: MachineConfig,
    ) -> None:
        """Rapid feed should be tool travel_feed_mm_s * 60."""
        ops: list[Operation] = [
            ToolUp(),
            RapidXY(x=50.0, y=50.0),
        ]
        gcode = gen.generate(ops)
        tc = config.get_tool("pen")
        expected_f = f"F{tc.travel_feed_mm_s * 60.0:.1f}"
        assert expected_f in gcode

    def test_draw_feed_in_mm_min(
        self, gen: GCodeGenerator, config: MachineConfig,
    ) -> None:
        """Draw feed should be tool feed_mm_s * 60."""
        ops: list[Operation] = [
            ToolUp(),
            RapidXY(x=50.0, y=50.0),
            ToolDown(),
            LinearMove(x=60.0, y=50.0),
            ToolUp(),
        ]
        gcode = gen.generate(ops)
        tc = config.get_tool("pen")
        expected_f = f"F{tc.feed_mm_s * 60.0:.1f}"
        assert expected_f in gcode

    def test_override_feed(self, gen: GCodeGenerator) -> None:
        """Explicit feed override should be converted to mm/min."""
        ops: list[Operation] = [
            ToolUp(),
            RapidXY(x=50.0, y=50.0),
            ToolDown(),
            LinearMove(x=60.0, y=50.0, feed=42.0),
            ToolUp(),
        ]
        gcode = gen.generate(ops)
        assert "F2520.0" in gcode  # 42 * 60


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


class TestCoordinateTransforms:
    def test_canvas_origin_mapped(
        self, gen: GCodeGenerator, config: MachineConfig,
    ) -> None:
        """Canvas (0, 0) should map to (offset_x, offset_y + height)."""
        ops: list[Operation] = [ToolUp(), RapidXY(x=0.0, y=0.0)]
        gcode = gen.generate(ops)
        # Canvas (0,0) = top-left -> machine (offset_x, offset_y + height)
        mx = config.canvas.offset_x_mm
        my = config.canvas.offset_y_mm + config.canvas.height_mm
        assert f"X{mx:.3f}" in gcode
        assert f"Y{my:.3f}" in gcode

    def test_no_g92_in_output(self, gen: GCodeGenerator) -> None:
        """G92 must never appear -- all coords are absolute machine."""
        ops: list[Operation] = [
            HomeXY(),
            ToolUp(),
            RapidXY(x=50.0, y=50.0),
            ToolDown(),
            DrawPolyline(points=((50.0, 50.0), (60.0, 60.0))),
            ToolUp(),
        ]
        gcode = gen.generate(ops)
        assert "G92" not in gcode


# ---------------------------------------------------------------------------
# Soft-limit validation
# ---------------------------------------------------------------------------


class TestSoftLimits:
    def test_reject_x_beyond_work_area(
        self, gen: GCodeGenerator, config: MachineConfig,
    ) -> None:
        # Canvas position that maps beyond work area X
        bad_x = config.work_area.x + 100.0
        ops: list[Operation] = [ToolUp(), RapidXY(x=bad_x, y=0.0)]
        with pytest.raises(GCodeError, match="X=.*outside work area"):
            gen.generate(ops)

    def test_reject_y_beyond_work_area(
        self, gen: GCodeGenerator, config: MachineConfig,
    ) -> None:
        bad_y = config.work_area.y + 100.0
        ops: list[Operation] = [ToolUp(), RapidXY(x=0.0, y=bad_y)]
        with pytest.raises(GCodeError):
            gen.generate(ops)


# ---------------------------------------------------------------------------
# Stroke generation (interactive mode)
# ---------------------------------------------------------------------------


class TestStrokeGeneration:
    def test_stroke_ends_with_m400(self, gen: GCodeGenerator) -> None:
        stroke = [
            RapidXY(x=50.0, y=50.0),
            ToolDown(),
            DrawPolyline(points=((50.0, 50.0), (60.0, 60.0))),
            ToolUp(),
        ]
        gcode = gen.generate_stroke(stroke)
        assert gcode.strip().endswith("M400")

    def test_stroke_contains_polyline_points(
        self, gen: GCodeGenerator,
    ) -> None:
        stroke = [
            RapidXY(x=50.0, y=50.0),
            ToolDown(),
            DrawPolyline(points=((50.0, 50.0), (60.0, 50.0), (60.0, 60.0))),
            ToolUp(),
        ]
        gcode = gen.generate_stroke(stroke)
        # Should have multiple G1 moves
        g1_count = gcode.count("G1 X")
        assert g1_count >= 3  # polyline points


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


class TestHome:
    def test_home_generates_g28(self, gen: GCodeGenerator) -> None:
        gcode = gen.generate([HomeXY(), ToolUp()])
        assert "G28 X Y" in gcode


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------


class TestToolSelection:
    def test_select_pen_macro(self, gen: GCodeGenerator) -> None:
        ops = [SelectTool(tool="pen"), ToolUp()]
        gcode = gen.generate(ops)
        assert "TOOL_PEN" in gcode

    def test_select_airbrush_macro(self, gen: GCodeGenerator) -> None:
        ops = [SelectTool(tool="airbrush"), ToolUp()]
        gcode = gen.generate(ops)
        assert "TOOL_AIRBRUSH" in gcode
