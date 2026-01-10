"""
Tests for G-code generator module.
"""

import pytest

from robot_control.configs.loader import (
    CanvasConfig,
    ConnectionConfig,
    FileExecutionConfig,
    InteractiveConfig,
    MachineConfig,
    MotionConfig,
    ToolConfig,
    WorkAreaConfig,
    ZStatesConfig,
)
from robot_control.gcode.generator import GCodeError, GCodeGenerator
from robot_control.job_ir.operations import (
    DrawPolyline,
    HomeXY,
    LinearMove,
    RapidXY,
    SelectTool,
    ToolDown,
    ToolUp,
)


@pytest.fixture
def config() -> MachineConfig:
    """Create test configuration."""
    return MachineConfig(
        connection=ConnectionConfig(
            socket_path="/tmp/klippy_uds",
            timeout_s=5.0,
            reconnect_attempts=3,
        ),
        work_area=WorkAreaConfig(x=250.0, y=350.0, z=40.0),
        canvas=CanvasConfig(
            offset_x_mm=25.0,
            offset_y_mm=25.0,
            width_mm=200.0,
            height_mm=280.0,
            y_flip=True,
        ),
        z_states=ZStatesConfig(
            travel_mm=10.0,
            pen_work_mm=20.0,
            airbrush_work_mm=0.0,
        ),
        tools={
            "pen": ToolConfig(
                xy_offset_mm=(0.0, 0.0),
                feed_mm_min=1500.0,
                travel_feed_mm_min=6000.0,
                plunge_feed_mm_min=300.0,
            ),
            "airbrush": ToolConfig(
                xy_offset_mm=(5.0, -3.0),
                feed_mm_min=3000.0,
                travel_feed_mm_min=6000.0,
                plunge_feed_mm_min=300.0,
                spray_height_mm=3.0,
            ),
        },
        motion=MotionConfig(
            max_velocity_mm_s=100.0,
            max_accel_mm_s2=1000.0,
            junction_deviation_mm=0.05,
        ),
        interactive=InteractiveConfig(
            jog_increments_mm=(0.1, 1.0, 10.0),
            default_jog_increment_mm=1.0,
            position_poll_interval_ms=100,
        ),
        file_execution=FileExecutionConfig(
            gcode_directory="/tmp/gcode",
        ),
    )


class TestGCodeGenerator:
    """Test G-code generation."""

    def test_header_generation(self, config: MachineConfig):
        """Generator should include initialization header."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([ToolUp()], include_header=True)

        assert "G21" in gcode  # Millimeters
        assert "G90" in gcode  # Absolute positioning

    def test_no_header_option(self, config: MachineConfig):
        """Generator should skip header when requested."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([ToolUp()], include_header=False)

        assert "G21" not in gcode
        assert "G90" not in gcode

    def test_home_xy(self, config: MachineConfig):
        """HomeXY should generate G28 command."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([HomeXY()], include_header=False)

        assert "G28 X Y" in gcode

    def test_tool_up(self, config: MachineConfig):
        """ToolUp should move to travel Z."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([ToolUp()], include_header=False)

        assert "Z10.000" in gcode  # travel_mm = 10.0
        assert "F300" in gcode  # plunge_feed

    def test_tool_down_pen(self, config: MachineConfig):
        """ToolDown with pen should move to pen work Z."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([SelectTool(tool="pen"), ToolDown()], include_header=False)

        assert "Z20.000" in gcode  # pen_work_mm = 20.0

    def test_tool_down_airbrush(self, config: MachineConfig):
        """ToolDown with airbrush should move to airbrush work Z."""
        gen = GCodeGenerator(config)
        gcode = gen.generate(
            [SelectTool(tool="airbrush"), ToolDown()],
            include_header=False,
        )

        assert "Z0.000" in gcode  # airbrush_work_mm = 0.0

    def test_rapid_xy_coordinate_transform(self, config: MachineConfig):
        """RapidXY should apply canvas and Y-flip transforms."""
        gen = GCodeGenerator(config)
        # Canvas origin (0, 0) with y_flip=True
        # Machine X = canvas_x + offset_x = 0 + 25 = 25
        # Machine Y = offset_y + (height - canvas_y) = 25 + (280 - 0) = 305
        gcode = gen.generate([RapidXY(x=0, y=0)], include_header=False)

        assert "X25.000" in gcode
        assert "Y305.000" in gcode
        assert "G0" in gcode

    def test_rapid_xy_with_tool_offset(self, config: MachineConfig):
        """RapidXY should apply tool offset."""
        gen = GCodeGenerator(config)
        # Airbrush has offset (5, -3)
        gcode = gen.generate(
            [SelectTool(tool="airbrush"), RapidXY(x=0, y=0)],
            include_header=False,
        )

        # X = 0 + 25 + 5 = 30
        # Y = 25 + 280 + (-3) = 302
        assert "X30.000" in gcode
        assert "Y302.000" in gcode

    def test_linear_move(self, config: MachineConfig):
        """LinearMove should use G1 command."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([LinearMove(x=10, y=10)], include_header=False)

        assert "G1" in gcode

    def test_linear_move_custom_feed(self, config: MachineConfig):
        """LinearMove should use custom feed if provided."""
        gen = GCodeGenerator(config)
        gcode = gen.generate(
            [LinearMove(x=10, y=10, feed=2500.0)],
            include_header=False,
        )

        assert "F2500" in gcode

    def test_draw_polyline(self, config: MachineConfig):
        """DrawPolyline should generate G1 for each point."""
        gen = GCodeGenerator(config)
        points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
        gcode = gen.generate([DrawPolyline(points=points)], include_header=False)

        # Should have G1 for each point
        g1_count = gcode.count("G1")
        assert g1_count == 3

    def test_select_tool_macro(self, config: MachineConfig):
        """SelectTool should call tool macro."""
        gen = GCodeGenerator(config)
        gcode = gen.generate([SelectTool(tool="pen")], include_header=False)
        assert "TOOL_PEN" in gcode

        gcode = gen.generate([SelectTool(tool="airbrush")], include_header=False)
        assert "TOOL_AIRBRUSH" in gcode


class TestGCodeLimitValidation:
    """Test soft limit validation."""

    def test_valid_position_passes(self, config: MachineConfig):
        """Position within bounds should pass validation."""
        gen = GCodeGenerator(config)
        # Should not raise
        gcode = gen.generate([RapidXY(x=100, y=140)], validate_limits=True)
        assert "G0" in gcode

    def test_position_exceeds_canvas_width(self, config: MachineConfig):
        """Drawing position exceeding canvas width should fail."""
        gen = GCodeGenerator(config)

        # Test position that exceeds canvas but not work area
        with pytest.raises(GCodeError, match="exceeds canvas"):
            gen.generate(
                [LinearMove(x=220, y=100)],  # canvas width is 200, but work area is 250
                validate_limits=True,
            )

    def test_position_exceeds_canvas_height(self, config: MachineConfig):
        """Drawing position exceeding canvas height should fail."""
        gen = GCodeGenerator(config)

        with pytest.raises(GCodeError, match="exceeds canvas"):
            gen.generate(
                [LinearMove(x=100, y=300)],  # canvas height is 280
                validate_limits=True,
            )

    def test_negative_position_fails(self, config: MachineConfig):
        """Negative canvas position should fail."""
        gen = GCodeGenerator(config)

        with pytest.raises(GCodeError, match="exceeds canvas"):
            gen.generate(
                [LinearMove(x=-10, y=100)],
                validate_limits=True,
            )

    def test_validation_can_be_disabled(self, config: MachineConfig):
        """Validation should be skippable."""
        gen = GCodeGenerator(config)
        # Should not raise even with invalid position
        gcode = gen.generate(
            [RapidXY(x=1000, y=1000)],
            validate_limits=False,
        )
        assert "G0" in gcode


class TestStrokeGeneration:
    """Test stroke G-code generation."""

    def test_generate_stroke_with_barrier(self, config: MachineConfig):
        """Stroke generation should include M400 barrier."""
        gen = GCodeGenerator(config)
        stroke = [
            RapidXY(x=10, y=10),
            ToolDown(),
            LinearMove(x=20, y=20),
            ToolUp(),
        ]

        gcode = gen.generate_stroke(stroke, include_barrier=True)
        assert "M400" in gcode

    def test_generate_stroke_without_barrier(self, config: MachineConfig):
        """Stroke generation can skip M400 barrier."""
        gen = GCodeGenerator(config)
        stroke = [RapidXY(x=10, y=10), ToolUp()]

        gcode = gen.generate_stroke(stroke, include_barrier=False)
        assert "M400" not in gcode
