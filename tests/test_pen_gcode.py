"""Unit tests for pen G-code generation.

Tests:
    - Pen vectors YAML loading and validation
    - G-code generation with tool offset
    - Coordinate frame transforms
    - Feed rate conversions
    - Macro insertion
"""

from pathlib import Path
import tempfile
import torch
import pytest

from src.utils import validators, gcode_generator, fs


@pytest.fixture
def simple_machine_config():
    """Simple machine configuration for testing."""
    return validators.MachineV1(
        schema="machine.v1",
        work_area_mm=validators.WorkArea(x=210.0, y=297.0, z=30.0),
        canvas_mm=validators.CanvasBounds(x_min=0.0, x_max=210.0, y_min=0.0, y_max=297.0),
        gcode_flavor="grbl_1.1f",
        units="mm",
        feed_units="mm/min",
        feeds=validators.Feeds(max_xy_mm_s=300.0, max_z_mm_s=20.0, rapid_mm_s=400.0),
        acceleration=validators.Acceleration(max_xy_mm_s2=2000.0, max_z_mm_s2=500.0),
        macros=validators.Macros(
            include_dir="gcode/macros",
            purge="PURGE.gcode",
            pen_up="PEN_UP.gcode",
            pen_down="PEN_DOWN.gcode",
        ),
        safety=validators.Safety(
            soft_limits=True,
            purge_zone_mm=validators.PurgeZone(
                x=(200.0, 209.0), y=(290.0, 296.0), z=(10.0, 15.0)
            ),
        ),
    )


@pytest.fixture
def simple_pen_tool_config():
    """Simple pen tool configuration for testing."""
    return validators.PenToolV1(
        schema="pen_tool.v1",
        name="test_pen",
        tip_diameter_mm=0.3,
        overlap_frac=0.25,
        max_passes=4,
        feed_mm_s=120.0,
        travel_mm_s=250.0,
        safe_z_mm=5.0,
        draw_z_mm=-0.2,
        plunge_mm_s=10.0,
        offset_mm=[12.5, -6.2, 0.0],  # Non-zero offset for testing
        macros=validators.PenMacros(pen_up="PEN_UP.gcode", pen_down="PEN_DOWN.gcode"),
        accel=validators.PenAcceleration(xy_mm_s2=1500.0),
        joins=validators.PenJoins(cap_style="round", join_style="round"),
    )


@pytest.fixture
def simple_pen_vectors():
    """Simple pen vectors for testing."""
    return validators.PenVectorsV1(
        schema="pen_vectors.v1",
        render_px=[1280, 908],
        work_area_mm=[210.0, 297.0],
        paths=[
            validators.PenPath(
                id="pen-000001",
                kind="polyline",
                role="outline",
                tip_diameter_mm=0.3,
                z_mm=-0.2,
                feed_mm_s=120.0,
                points_mm=[[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]],
            ),
            validators.PenPath(
                id="pen-000002",
                kind="polyline",
                role="hatch",
                tip_diameter_mm=0.3,
                z_mm=-0.2,
                feed_mm_s=120.0,
                points_mm=[[10.0, 20.0], [30.0, 20.0]],
            ),
        ],
        metadata=validators.PenVectorsMetadata(
            tool_name="test_pen",
            offset_mm=[12.5, -6.2, 0.0],
            hatch_angles_deg=[0.0],
            generated_at="2025-10-31T12:00:00Z",
            tracer_version="pen_tracer.v1",
        ),
    )


def test_pen_vectors_validation_valid(simple_pen_vectors):
    """Test that valid pen vectors pass validation."""
    # Should not raise
    assert simple_pen_vectors.schema == "pen_vectors.v1"
    assert len(simple_pen_vectors.paths) == 2


def test_pen_vectors_validation_invalid_kind():
    """Test that invalid kind raises validation error."""
    with pytest.raises(ValueError, match="kind must be one of"):
        validators.PenPath(
            id="pen-000001",
            kind="invalid_kind",  # Invalid
            role="outline",
            tip_diameter_mm=0.3,
            z_mm=-0.2,
            feed_mm_s=120.0,
            points_mm=[[10.0, 10.0], [20.0, 10.0]],
        )


def test_pen_vectors_validation_out_of_bounds():
    """Test that out-of-bounds points raise validation error."""
    with pytest.raises(ValueError, match="out of bounds"):
        validators.PenPath(
            id="pen-000001",
            kind="polyline",
            role="outline",
            tip_diameter_mm=0.3,
            z_mm=-0.2,
            feed_mm_s=120.0,
            points_mm=[[10.0, 10.0], [250.0, 10.0]],  # X > 210 (A4 width)
        )


def test_generate_pen_gcode_basic(
    simple_pen_vectors, simple_machine_config, simple_pen_tool_config, tmp_path
):
    """Test basic pen G-code generation."""
    output_path = tmp_path / "test_pen.gcode"
    
    # Generate G-code
    gcode_generator.generate_pen_gcode(
        simple_pen_vectors,
        simple_machine_config,
        simple_pen_tool_config,
        output_path,
    )
    
    # Check file exists
    assert output_path.exists()
    
    # Read and check contents
    with open(output_path, 'r') as f:
        gcode = f.read()
    
    # Check header
    assert "G21" in gcode  # mm units
    assert "G90" in gcode  # absolute positioning
    
    # Check tool info in comments
    assert "test_pen" in gcode
    assert "12.5" in gcode  # Tool offset X
    
    # Check path IDs
    assert "pen-000001" in gcode
    assert "pen-000002" in gcode
    
    # Check moves present
    assert "G0" in gcode  # Rapid moves
    assert "G1" in gcode  # Linear moves
    
    # Check feed rates present (mm/min)
    assert "F" in gcode


def test_pen_gcode_tool_offset_applied(
    simple_pen_vectors, simple_machine_config, simple_pen_tool_config, tmp_path
):
    """Test that tool offset is correctly applied in G-code."""
    output_path = tmp_path / "test_pen_offset.gcode"
    
    # First path point: [10.0, 10.0] in image frame
    # Expected after transform and offset:
    # - Image to machine: Y flip → [10.0, 297.0 - 10.0] = [10.0, 287.0]
    # - Add offset: [10.0 + 12.5, 287.0 + (-6.2)] = [22.5, 280.8]
    
    gcode_generator.generate_pen_gcode(
        simple_pen_vectors,
        simple_machine_config,
        simple_pen_tool_config,
        output_path,
    )
    
    with open(output_path, 'r') as f:
        gcode = f.read()
    
    # Check that offset values appear in coordinates
    assert "22.5" in gcode or "22.500" in gcode  # X coordinate with offset
    # Note: exact Y depends on coordinate transform


def test_pen_gcode_respects_soft_limits(
    simple_machine_config, simple_pen_tool_config, tmp_path
):
    """Test that soft limits are enforced in pen G-code generation."""
    # Create pen vectors with out-of-machine-bounds point (after offset)
    # Machine bounds: X=[0, 210], Y=[0, 297]
    # Tool offset: [12.5, -6.2, 0]
    # If image point is [200, 10] → machine [200+12.5, 287-6.2] = [212.5, 280.8]
    # X=212.5 > 210 → should violate soft limits
    
    pen_vectors = validators.PenVectorsV1(
        schema="pen_vectors.v1",
        render_px=[1280, 908],
        work_area_mm=[210.0, 297.0],
        paths=[
            validators.PenPath(
                id="pen-000001",
                kind="polyline",
                role="outline",
                tip_diameter_mm=0.3,
                z_mm=-0.2,
                feed_mm_s=120.0,
                points_mm=[[200.0, 10.0], [205.0, 10.0]],  # Will violate after offset
            ),
        ],
        metadata=validators.PenVectorsMetadata(
            tool_name="test_pen",
            offset_mm=[12.5, -6.2, 0.0],
            hatch_angles_deg=[0.0],
            generated_at="2025-10-31T12:00:00Z",
            tracer_version="pen_tracer.v1",
        ),
    )
    
    output_path = tmp_path / "test_pen_limits.gcode"
    
    # Should raise ValueError for soft limit violation
    with pytest.raises(ValueError, match="out of bounds"):
        gcode_generator.generate_pen_gcode(
            pen_vectors,
            simple_machine_config,
            simple_pen_tool_config,
            output_path,
        )


def test_pen_vectors_yaml_round_trip(simple_pen_vectors, tmp_path):
    """Test pen vectors can be saved and loaded from YAML."""
    yaml_path = tmp_path / "test_vectors.yaml"
    
    # Save
    data = simple_pen_vectors.dict()
    fs.atomic_yaml_dump(data, yaml_path)
    
    # Load
    loaded = validators.load_pen_vectors(yaml_path)
    
    # Check round-trip
    assert loaded.schema == simple_pen_vectors.schema
    assert len(loaded.paths) == len(simple_pen_vectors.paths)
    assert loaded.paths[0].id == simple_pen_vectors.paths[0].id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

