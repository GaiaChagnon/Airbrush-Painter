#!/usr/bin/env python3
"""Round-trip validation tests for all schema types in configs/schema/.

Tests:
- Each schema YAML file loads and parses correctly
- Pydantic validators accept valid data and reject invalid data
- Round-trip: construct model -> dict -> reconstruct model
- Boundary values trigger correct validation errors
- File-based loaders raise on missing files and invalid content

Run with: pytest tests/test_schemas.py -v
"""

from pathlib import Path

import pytest

from src.utils import fs, validators

_SCHEMA_DIR = Path(__file__).parent.parent / "configs" / "schema"


# ============================================================================
# SCHEMA YAML LOADING (all schema files parse as valid YAML)
# ============================================================================

_SCHEMA_FILES = list(_SCHEMA_DIR.glob("*.yaml"))


@pytest.mark.parametrize(
    "schema_file",
    _SCHEMA_FILES,
    ids=[f.stem for f in _SCHEMA_FILES],
)
def test_schema_yaml_loads(schema_file: Path):
    """Each schema YAML file must load without errors."""
    data = fs.load_yaml(schema_file)
    assert isinstance(data, dict), (
        f"Expected dict from {schema_file.name}, got {type(data).__name__}"
    )
    assert "schema" in data, (
        f"Schema file {schema_file.name} missing 'schema' key"
    )


# ============================================================================
# STROKE SCHEMA V1
# ============================================================================

def _valid_stroke_data(**overrides):
    """Return minimal valid StrokeV1 data dict, with optional overrides."""
    base = {
        "id": "00001-00001-abcdef12",
        "bezier": {
            "p1": [10.0, 10.0],
            "p2": [50.0, 20.0],
            "p3": [100.0, 80.0],
            "p4": [150.0, 100.0],
        },
        "z_profile": {"z0": 5.0, "z1": 12.0},
        "speed_profile": {"v0": 50.0, "v1": 200.0},
        "color_cmy": {"c": 0.0, "m": 0.5, "y": 1.0},
    }
    base.update(overrides)
    return base


class TestStrokeSchemaV1:
    """Stroke schema validation tests."""

    def test_valid_stroke_round_trip(self):
        """Construct -> dict -> reconstruct preserves data."""
        data = _valid_stroke_data()
        stroke = validators.StrokeV1(**data)
        d = stroke.dict()
        stroke2 = validators.StrokeV1(**d)
        assert stroke2.id == stroke.id
        assert stroke2.bezier.p1 == stroke.bezier.p1

    def test_boundary_z_zero(self):
        """z0=0.0 is at the lower boundary and should be accepted."""
        data = _valid_stroke_data()
        data["z_profile"] = {"z0": 0.0, "z1": 0.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.z_profile.z0 == 0.0

    def test_boundary_z_max(self):
        """z=30.0 is at the upper boundary and should be accepted."""
        data = _valid_stroke_data()
        data["z_profile"] = {"z0": 30.0, "z1": 30.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.z_profile.z1 == 30.0

    def test_boundary_speed_min(self):
        """v=1.0 is at the lower boundary and should be accepted."""
        data = _valid_stroke_data()
        data["speed_profile"] = {"v0": 1.0, "v1": 1.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.speed_profile.v0 == 1.0

    def test_boundary_speed_max(self):
        """v=300.0 is at the upper boundary and should be accepted."""
        data = _valid_stroke_data()
        data["speed_profile"] = {"v0": 300.0, "v1": 300.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.speed_profile.v1 == 300.0

    def test_boundary_cmy_zero(self):
        """CMY=0.0 at the lower boundary."""
        data = _valid_stroke_data()
        data["color_cmy"] = {"c": 0.0, "m": 0.0, "y": 0.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.color_cmy.c == 0.0

    def test_boundary_cmy_one(self):
        """CMY=1.0 at the upper boundary."""
        data = _valid_stroke_data()
        data["color_cmy"] = {"c": 1.0, "m": 1.0, "y": 1.0}
        stroke = validators.StrokeV1(**data)
        assert stroke.color_cmy.y == 1.0

    def test_invalid_cmy_negative(self):
        """CMY < 0 must be rejected."""
        data = _valid_stroke_data()
        data["color_cmy"] = {"c": -0.1, "m": 0.0, "y": 0.0}
        with pytest.raises(ValueError):
            validators.StrokeV1(**data)

    def test_invalid_speed_above_max(self):
        """Speed > 300 must be rejected."""
        data = _valid_stroke_data()
        data["speed_profile"] = {"v0": 301.0, "v1": 100.0}
        with pytest.raises(ValueError):
            validators.StrokeV1(**data)

    def test_invalid_bezier_x_negative(self):
        """Negative x coordinate must be rejected."""
        data = _valid_stroke_data()
        data["bezier"]["p1"] = [-1.0, 10.0]
        with pytest.raises(ValueError, match="out of bounds"):
            validators.StrokeV1(**data)

    def test_invalid_bezier_y_above_a4(self):
        """y > 297 (A4 height) must be rejected."""
        data = _valid_stroke_data()
        data["bezier"]["p3"] = [100.0, 300.0]
        with pytest.raises(ValueError, match="out of bounds"):
            validators.StrokeV1(**data)

    def test_missing_bezier_field(self):
        """Missing control point must raise."""
        data = _valid_stroke_data()
        del data["bezier"]["p4"]
        with pytest.raises(ValueError):
            validators.StrokeV1(**data)

    def test_strokes_file_wrong_schema_version(self):
        """Wrong schema version string must be rejected."""
        data = {
            "schema": "stroke.v2",
            "strokes": [_valid_stroke_data()],
        }
        with pytest.raises(ValueError, match="stroke.v1"):
            validators.StrokesFileV1(**data)


# ============================================================================
# LUT SCHEMA V1
# ============================================================================

def _valid_lut_data(**overrides):
    """Return minimal valid LUTsV1 data dict."""
    base = {
        "schema": "luts.v1",
        "semver": "1.0.0",
        "color_lut": {
            "grid": {"c": 5, "m": 5, "y": 5},
            "domain": [0.0, 1.0],
            "shape": [5, 5, 5, 3],
            "dtype": "float32",
            "values_path": "luts/color.pt",
        },
        "alpha_lut": {
            "axes": {"z": [5.0, 10.0, 15.0], "v": [50.0, 100.0]},
            "shape": [3, 2],
            "dtype": "float32",
            "values_path": "luts/alpha.pt",
        },
        "psf_lut": {
            "axes": {"z": [5.0, 10.0], "v": [50.0, 100.0]},
            "kernel_size": 7,
            "shape": [2, 2, 7, 7],
            "dtype": "float32",
            "values_path": "luts/psf.pt",
        },
        "metadata": {
            "machine_profile": "test_machine",
            "calibration_date": "2025-01-01",
            "operator": "test",
        },
    }
    base.update(overrides)
    return base


class TestLUTSchemaV1:
    """LUT schema validation tests."""

    def test_valid_lut_round_trip(self):
        """Construct -> dict -> reconstruct preserves data."""
        data = _valid_lut_data()
        lut = validators.LUTsV1(**data)
        d = lut.dict()
        lut2 = validators.LUTsV1(**d)
        assert lut2.color_lut.grid.c == lut.color_lut.grid.c
        assert lut2.psf_lut.kernel_size == lut.psf_lut.kernel_size

    def test_wrong_schema_version(self):
        """Wrong schema version must be rejected."""
        data = _valid_lut_data(schema="luts.v2")
        with pytest.raises(ValueError, match="luts.v1"):
            validators.LUTsV1(**data)

    def test_color_lut_shape_mismatch(self):
        """Shape inconsistent with grid dimensions must be rejected."""
        data = _valid_lut_data()
        data["color_lut"]["shape"] = [5, 5, 3, 3]  # Wrong: should be [5,5,5,3]
        with pytest.raises(ValueError, match="Shape"):
            validators.LUTsV1(**data)

    def test_alpha_lut_shape_mismatch(self):
        """Alpha LUT shape inconsistent with axes must be rejected."""
        data = _valid_lut_data()
        data["alpha_lut"]["shape"] = [4, 2]  # Wrong: axes.z has 3 elements
        with pytest.raises(ValueError, match="Shape"):
            validators.LUTsV1(**data)

    def test_psf_even_kernel_rejected(self):
        """Even kernel size must be rejected."""
        data = _valid_lut_data()
        data["psf_lut"]["kernel_size"] = 8
        data["psf_lut"]["shape"] = [2, 2, 8, 8]
        with pytest.raises(ValueError, match="odd"):
            validators.LUTsV1(**data)

    def test_psf_shape_mismatch(self):
        """PSF shape inconsistent with axes and kernel must be rejected."""
        data = _valid_lut_data()
        data["psf_lut"]["shape"] = [2, 2, 5, 5]  # kernel_size is 7
        with pytest.raises(ValueError, match="Shape"):
            validators.LUTsV1(**data)

    def test_grid_minimum(self):
        """Grid dimension < 2 must be rejected."""
        data = _valid_lut_data()
        data["color_lut"]["grid"]["c"] = 1
        with pytest.raises(ValueError):
            validators.LUTsV1(**data)


# ============================================================================
# MACHINE SCHEMA V1
# ============================================================================

_VALID_MACHINE_DATA = {
    "schema": "machine.v1",
    "work_area_mm": {"x": 250.0, "y": 350.0, "z": 40.0},
    "canvas_mm": {"x_min": 20.0, "x_max": 230.0, "y_min": 26.5, "y_max": 323.5},
    "gcode_flavor": "grbl_1.1f",
    "units": "mm",
    "feed_units": "mm/min",
    "feeds": {"max_xy_mm_s": 200.0, "max_z_mm_s": 30.0, "rapid_mm_s": 300.0},
    "acceleration": {"max_xy_mm_s2": 1000.0, "max_z_mm_s2": 500.0},
    "macros": {
        "include_dir": "macros/",
        "purge": "purge.g",
        "pen_up": "pen_up.g",
        "pen_down": "pen_down.g",
    },
    "safety": {
        "soft_limits": True,
        "purge_zone_mm": {
            "x": [0.0, 20.0],
            "y": [0.0, 26.5],
            "z": [0.0, 40.0],
        },
    },
}


class TestMachineSchemaV1:
    """Machine schema validation tests."""

    def test_valid_machine_round_trip(self):
        """Construct -> dict -> reconstruct."""
        cfg = validators.MachineV1(**_VALID_MACHINE_DATA)
        d = cfg.dict()
        cfg2 = validators.MachineV1(**d)
        assert cfg2.work_area_mm.x == cfg.work_area_mm.x

    def test_wrong_schema_version(self):
        """Wrong schema version must be rejected."""
        data = {**_VALID_MACHINE_DATA, "schema": "machine.v2"}
        with pytest.raises(ValueError, match="machine.v1"):
            validators.MachineV1(**data)

    def test_invalid_gcode_flavor(self):
        """Unknown G-code flavor must be rejected."""
        data = {**_VALID_MACHINE_DATA, "gcode_flavor": "unknown_flavor"}
        with pytest.raises(ValueError, match="G-code flavor"):
            validators.MachineV1(**data)

    def test_invalid_units(self):
        """Invalid unit string must be rejected."""
        data = {**_VALID_MACHINE_DATA, "units": "meters"}
        with pytest.raises(ValueError, match="Units"):
            validators.MachineV1(**data)

    def test_invalid_feed_units(self):
        """Invalid feed unit string must be rejected."""
        data = {**_VALID_MACHINE_DATA, "feed_units": "inches/s"}
        with pytest.raises(ValueError, match="Feed units"):
            validators.MachineV1(**data)

    def test_canvas_exceeds_work_area(self):
        """Canvas x_max beyond work area must be rejected."""
        data = {**_VALID_MACHINE_DATA}
        data["canvas_mm"] = {
            "x_min": 20.0, "x_max": 300.0,
            "y_min": 26.5, "y_max": 323.5,
        }
        with pytest.raises(ValueError, match="exceeds"):
            validators.MachineV1(**data)

    def test_zero_work_area_rejected(self):
        """Work area dimension of 0 must be rejected."""
        data = {**_VALID_MACHINE_DATA}
        data["work_area_mm"] = {"x": 0.0, "y": 350.0, "z": 40.0}
        with pytest.raises(ValueError):
            validators.MachineV1(**data)

    def test_negative_feed_rate_rejected(self):
        """Negative feed rate must be rejected."""
        data = {**_VALID_MACHINE_DATA}
        data["feeds"] = {
            "max_xy_mm_s": -10.0,
            "max_z_mm_s": 30.0,
            "rapid_mm_s": 300.0,
        }
        with pytest.raises(ValueError):
            validators.MachineV1(**data)

    def test_load_machine_profile_file_not_found(self):
        """Loading from nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validators.load_machine_profile("/tmp/nonexistent_machine.yaml")

    def test_load_machine_profile_invalid_content(self, tmp_path):
        """Loading invalid YAML content must raise ValueError."""
        bad_file = tmp_path / "bad_machine.yaml"
        fs.atomic_yaml_dump({"schema": "machine.v1", "junk": True}, bad_file)
        with pytest.raises(ValueError, match="validation failed"):
            validators.load_machine_profile(bad_file)

    def test_load_machine_profile_round_trip(self, tmp_path):
        """Write valid config -> load -> verify fields."""
        cfg_path = tmp_path / "machine.yaml"
        fs.atomic_yaml_dump(_VALID_MACHINE_DATA, cfg_path)
        cfg = validators.load_machine_profile(cfg_path)
        assert cfg.gcode_flavor == "grbl_1.1f"
        assert cfg.canvas_mm.x_min == 20.0

    def test_marlin_flavor_accepted(self):
        """marlin_2.0 flavor must be accepted."""
        data = {**_VALID_MACHINE_DATA, "gcode_flavor": "marlin_2.0"}
        cfg = validators.MachineV1(**data)
        assert cfg.gcode_flavor == "marlin_2.0"

    def test_inch_units_accepted(self):
        """inch units must be accepted."""
        data = {**_VALID_MACHINE_DATA, "units": "inch"}
        cfg = validators.MachineV1(**data)
        assert cfg.units == "inch"


# ============================================================================
# JOB SCHEMA V1
# ============================================================================

def _valid_job_data(**overrides):
    """Return minimal valid JobV1 data dict."""
    base = {
        "schema": "job.v1",
        "machine_profile": "grbl_airbrush.v1",
        "inputs": {
            "target_image_path": "data/target.png",
            "pen_layer_path": None,
        },
        "limits": {
            "max_strokes": 1500,
            "bbox_mm": {
                "x": [0.0, 210.0],
                "y": [0.0, 297.0],
                "z": [0.0, 30.0],
            },
        },
        "passes": [
            {"name": "cmy", "strokes_path": "strokes.yaml"},
        ],
        "artifacts": {
            "cmy_gcode_out": "output/cmy.gcode",
            "pen_gcode_out": None,
            "manifest_out": "output/manifest.yaml",
        },
    }
    base.update(overrides)
    return base


class TestJobSchemaV1:
    """Job schema validation tests."""

    def test_valid_job_round_trip(self):
        """Construct -> dict -> reconstruct."""
        data = _valid_job_data()
        job = validators.JobV1(**data)
        d = job.dict()
        job2 = validators.JobV1(**d)
        assert job2.machine_profile == job.machine_profile
        assert len(job2.passes) == 1

    def test_wrong_schema_version(self):
        """Wrong schema version must be rejected."""
        data = _valid_job_data(schema="job.v2")
        with pytest.raises(ValueError, match="job.v1"):
            validators.JobV1(**data)

    def test_invalid_pass_name(self):
        """Pass name other than 'cmy' or 'pen' must be rejected."""
        data = _valid_job_data()
        data["passes"] = [{"name": "watercolor", "strokes_path": "x.yaml"}]
        with pytest.raises(ValueError, match="Pass name"):
            validators.JobV1(**data)

    def test_zero_max_strokes_rejected(self):
        """max_strokes=0 must be rejected (gt=0)."""
        data = _valid_job_data()
        data["limits"]["max_strokes"] = 0
        with pytest.raises(ValueError):
            validators.JobV1(**data)

    def test_multiple_passes(self):
        """Job with both cmy and pen passes must be accepted."""
        data = _valid_job_data()
        data["passes"] = [
            {"name": "cmy", "strokes_path": "strokes.yaml"},
            {"name": "pen", "vectors_path": "pen.yaml"},
        ]
        job = validators.JobV1(**data)
        assert len(job.passes) == 2

    def test_load_job_file_not_found(self):
        """Loading from nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validators.load_job_config("/tmp/nonexistent_job.yaml")

    def test_load_job_invalid_content(self, tmp_path):
        """Loading invalid YAML content must raise ValueError."""
        bad_file = tmp_path / "bad_job.yaml"
        fs.atomic_yaml_dump({"schema": "job.v1", "junk": True}, bad_file)
        with pytest.raises(ValueError, match="validation failed"):
            validators.load_job_config(bad_file)

    def test_load_job_round_trip(self, tmp_path):
        """Write -> load -> verify."""
        cfg_path = tmp_path / "job.yaml"
        fs.atomic_yaml_dump(_valid_job_data(), cfg_path)
        job = validators.load_job_config(cfg_path)
        assert job.machine_profile == "grbl_airbrush.v1"


# ============================================================================
# PEN TOOL SCHEMA V1
# ============================================================================

def _valid_pen_tool_data(**overrides):
    """Return minimal valid PenToolV1 data dict."""
    base = {
        "schema": "pen_tool.v1",
        "name": "finetip_pigment_0.3mm",
        "tip_diameter_mm": 0.30,
        "overlap_frac": 0.25,
        "max_passes": 4,
        "feed_mm_s": 120.0,
        "travel_mm_s": 250.0,
        "safe_z_mm": 5.0,
        "draw_z_mm": -0.2,
        "plunge_mm_s": 10.0,
        "offset_mm": [12.50, -6.20, 0.00],
        "macros": {
            "pen_up": "PEN_UP.gcode",
            "pen_down": "PEN_DOWN.gcode",
        },
        "accel": {"xy_mm_s2": 1500.0},
        "joins": {"cap_style": "round", "join_style": "round"},
    }
    base.update(overrides)
    return base


class TestPenToolSchemaV1:
    """Pen tool schema validation tests."""

    def test_valid_pen_tool_round_trip(self):
        """Construct -> dict -> reconstruct."""
        data = _valid_pen_tool_data()
        pen = validators.PenToolV1(**data)
        d = pen.dict()
        pen2 = validators.PenToolV1(**d)
        assert pen2.name == pen.name
        assert pen2.tip_diameter_mm == pen.tip_diameter_mm

    def test_wrong_schema_version(self):
        """Wrong schema version must be rejected."""
        data = _valid_pen_tool_data(schema="pen_tool.v2")
        with pytest.raises(ValueError, match="pen_tool.v1"):
            validators.PenToolV1(**data)

    def test_tip_diameter_below_min(self):
        """tip_diameter_mm < 0.1 must be rejected."""
        data = _valid_pen_tool_data(tip_diameter_mm=0.05)
        with pytest.raises(ValueError):
            validators.PenToolV1(**data)

    def test_tip_diameter_above_max(self):
        """tip_diameter_mm > 2.0 must be rejected."""
        data = _valid_pen_tool_data(tip_diameter_mm=3.0)
        with pytest.raises(ValueError):
            validators.PenToolV1(**data)

    def test_offset_wrong_length(self):
        """offset_mm with != 3 elements must be rejected."""
        data = _valid_pen_tool_data(offset_mm=[1.0, 2.0])
        with pytest.raises(ValueError, match="3 elements"):
            validators.PenToolV1(**data)

    def test_offset_out_of_bounds(self):
        """offset_mm element > 50 must be rejected."""
        data = _valid_pen_tool_data(offset_mm=[60.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="out of bounds"):
            validators.PenToolV1(**data)

    def test_invalid_cap_style(self):
        """Unknown cap style must be rejected."""
        data = _valid_pen_tool_data()
        data["joins"] = {"cap_style": "diamond", "join_style": "round"}
        with pytest.raises(ValueError, match="cap_style"):
            validators.PenToolV1(**data)

    def test_invalid_join_style(self):
        """Unknown join style must be rejected."""
        data = _valid_pen_tool_data()
        data["joins"] = {"cap_style": "round", "join_style": "chamfer"}
        with pytest.raises(ValueError, match="join_style"):
            validators.PenToolV1(**data)

    def test_load_pen_tool_file_not_found(self):
        """Loading from nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validators.load_pen_tool_config("/tmp/nonexistent_pen.yaml")

    def test_boundary_draw_z_negative(self):
        """draw_z_mm at -2.0 (min boundary) must be accepted."""
        data = _valid_pen_tool_data(draw_z_mm=-2.0)
        pen = validators.PenToolV1(**data)
        assert pen.draw_z_mm == -2.0

    def test_boundary_overlap_frac_zero(self):
        """overlap_frac=0.0 boundary must be accepted."""
        data = _valid_pen_tool_data(overlap_frac=0.0)
        pen = validators.PenToolV1(**data)
        assert pen.overlap_frac == 0.0


# ============================================================================
# PEN VECTORS SCHEMA V1
# ============================================================================

def _valid_pen_vectors_data(**overrides):
    """Return minimal valid PenVectorsV1 data dict."""
    base = {
        "schema": "pen_vectors.v1",
        "render_px": [1280, 908],
        "work_area_mm": [210.0, 297.0],
        "paths": [
            {
                "id": "pen-000001",
                "kind": "polyline",
                "role": "outline",
                "tip_diameter_mm": 0.30,
                "z_mm": -0.2,
                "feed_mm_s": 120.0,
                "points_mm": [[10.0, 10.0], [50.0, 50.0], [100.0, 100.0]],
            },
        ],
        "metadata": {
            "tool_name": "finetip_pigment_0.3mm",
            "offset_mm": [12.50, -6.20, 0.00],
            "hatch_angles_deg": [0.0, 90.0, 45.0, 135.0],
            "generated_at": "2025-10-31T12:00:00Z",
            "tracer_version": "pen_tracer.v2",
        },
    }
    base.update(overrides)
    return base


class TestPenVectorsSchemaV1:
    """Pen vectors schema validation tests."""

    def test_valid_pen_vectors_round_trip(self):
        """Construct -> dict -> reconstruct."""
        data = _valid_pen_vectors_data()
        pv = validators.PenVectorsV1(**data)
        d = pv.dict()
        pv2 = validators.PenVectorsV1(**d)
        assert len(pv2.paths) == len(pv.paths)
        assert pv2.metadata.tool_name == pv.metadata.tool_name

    def test_wrong_schema_version(self):
        """Wrong schema version must be rejected."""
        data = _valid_pen_vectors_data(schema="pen_vectors.v2")
        with pytest.raises(ValueError, match="pen_vectors.v1"):
            validators.PenVectorsV1(**data)

    def test_invalid_path_kind(self):
        """Unknown path kind must be rejected."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["kind"] = "spline"
        with pytest.raises(ValueError, match="kind"):
            validators.PenVectorsV1(**data)

    def test_invalid_path_role(self):
        """Unknown path role must be rejected."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["role"] = "decoration"
        with pytest.raises(ValueError, match="role"):
            validators.PenVectorsV1(**data)

    def test_point_out_of_bounds_x(self):
        """Point x > 210 must be rejected."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["points_mm"] = [[220.0, 10.0], [50.0, 50.0]]
        with pytest.raises(ValueError, match="out of bounds"):
            validators.PenVectorsV1(**data)

    def test_point_out_of_bounds_y(self):
        """Point y > 297 must be rejected."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["points_mm"] = [[10.0, 300.0], [50.0, 50.0]]
        with pytest.raises(ValueError, match="out of bounds"):
            validators.PenVectorsV1(**data)

    def test_point_wrong_dimensions(self):
        """Point with != 2 coordinates must be rejected."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["points_mm"] = [[10.0, 20.0, 30.0]]
        with pytest.raises(ValueError, match="2 coordinates"):
            validators.PenVectorsV1(**data)

    def test_render_px_wrong_length(self):
        """render_px with != 2 elements must be rejected."""
        data = _valid_pen_vectors_data(render_px=[1280])
        with pytest.raises(ValueError, match="2 elements"):
            validators.PenVectorsV1(**data)

    def test_render_px_non_positive(self):
        """render_px with 0 dimension must be rejected."""
        data = _valid_pen_vectors_data(render_px=[0, 908])
        with pytest.raises(ValueError, match="positive"):
            validators.PenVectorsV1(**data)

    def test_work_area_non_positive(self):
        """work_area_mm with 0 dimension must be rejected."""
        data = _valid_pen_vectors_data(work_area_mm=[0.0, 297.0])
        with pytest.raises(ValueError, match="positive"):
            validators.PenVectorsV1(**data)

    def test_polygon_kind_accepted(self):
        """'polygon' kind must be accepted."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["kind"] = "polygon"
        pv = validators.PenVectorsV1(**data)
        assert pv.paths[0].kind == "polygon"

    def test_hatch_role_accepted(self):
        """'hatch' role must be accepted."""
        data = _valid_pen_vectors_data()
        data["paths"][0]["role"] = "hatch"
        pv = validators.PenVectorsV1(**data)
        assert pv.paths[0].role == "hatch"


# ============================================================================
# FLATTEN CONFIG FOR MLFLOW
# ============================================================================

class TestFlattenConfig:
    """Tests for flatten_config_for_mlflow."""

    def test_flat_dict_passthrough(self):
        """Already flat dict should pass through unchanged."""
        d = {"a": 1, "b": "hello"}
        flat = validators.flatten_config_for_mlflow(d)
        assert flat == d

    def test_nested_dict_dot_separated(self):
        """Nested keys should become dot-separated."""
        d = {"outer": {"inner": 42}}
        flat = validators.flatten_config_for_mlflow(d)
        assert flat["outer.inner"] == 42

    def test_list_values_json_encoded(self):
        """List values should be JSON-encoded strings."""
        import json
        d = {"items": [1, 2, 3]}
        flat = validators.flatten_config_for_mlflow(d)
        assert flat["items"] == json.dumps([1, 2, 3])

    def test_pydantic_model_input(self):
        """Should accept pydantic BaseModel instances."""
        data = _valid_stroke_data()
        stroke = validators.StrokeV1(**data)
        flat = validators.flatten_config_for_mlflow(stroke)
        assert "id" in flat
        assert "bezier.p1" in flat


# ============================================================================
# FILE-BASED LOADER ERROR PATHS
# ============================================================================

class TestLoaderErrorPaths:
    """Test error paths for file-based config loaders."""

    def test_load_luts_file_not_found(self):
        """Loading LUTs from nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validators.load_luts_config("/tmp/nonexistent_luts.yaml")

    def test_load_strokes_file_not_found(self):
        """Loading strokes from nonexistent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validators.validate_strokes_file("/tmp/nonexistent_strokes.yaml")

    def test_load_luts_invalid_content(self, tmp_path):
        """Loading invalid LUT YAML must raise ValueError."""
        bad_file = tmp_path / "bad_luts.yaml"
        fs.atomic_yaml_dump({"schema": "luts.v1", "junk": True}, bad_file)
        with pytest.raises(ValueError, match="validation failed"):
            validators.load_luts_config(bad_file)

    def test_load_strokes_invalid_content(self, tmp_path):
        """Loading invalid strokes YAML must raise ValueError."""
        bad_file = tmp_path / "bad_strokes.yaml"
        fs.atomic_yaml_dump({"schema": "stroke.v1", "junk": True}, bad_file)
        with pytest.raises(ValueError, match="validation failed"):
            validators.validate_strokes_file(bad_file)

    def test_load_strokes_valid_round_trip(self, tmp_path):
        """Write valid strokes -> load -> verify."""
        strokes_data = {
            "schema": "stroke.v1",
            "strokes": [_valid_stroke_data()],
        }
        cfg_path = tmp_path / "strokes.yaml"
        fs.atomic_yaml_dump(strokes_data, cfg_path)
        result = validators.validate_strokes_file(cfg_path)
        assert len(result.strokes) == 1
        assert result.strokes[0].id == "00001-00001-abcdef12"
