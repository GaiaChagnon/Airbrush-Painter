#!/usr/bin/env python3
"""Edge case and boundary value tests for utils modules.

Covers gaps not addressed in test_utils_comprehensive.py:
- mm_to_px / px_to_mm at zero and boundary values
- atomic_yaml_dump content validity
- Geometry functions with degenerate inputs
- fs module edge cases
- Compute module edge cases

Run with: pytest tests/test_edge_cases.py -v
"""

from pathlib import Path

import pytest
import torch
import yaml

from src.utils import (
    color,
    compute,
    fs,
    geometry,
    hashing,
    strokes,
    validators,
)


# ============================================================================
# MM <-> PX ZERO AND BOUNDARY EDGE CASES
# ============================================================================

class TestMmPxZeroEdgeCases:
    """Test mm_to_px and px_to_mm at zero and boundary values."""

    _WORK_AREA = (210.0, 297.0)
    _RENDER_PX = (908, 1280)

    def test_mm_to_px_zero_coordinate(self):
        """mm_to_px(0, 0) should map to a valid pixel coordinate."""
        x_mm = torch.tensor([[0.0, 0.0]])
        x_px = compute.mm_to_px(
            x_mm, self._WORK_AREA, self._RENDER_PX, image_origin="top_left"
        )
        assert x_px.shape == (1, 2)
        # X=0mm should map to X=0px
        assert x_px[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        # Y=0mm with top_left origin -> Y = height_px (bottom of image)
        assert x_px[0, 1].item() == pytest.approx(1280.0, abs=1e-5)

    def test_px_to_mm_zero_coordinate(self):
        """px_to_mm(0, 0) should map back to a valid mm coordinate."""
        x_px = torch.tensor([[0.0, 0.0]])
        x_mm = compute.px_to_mm(
            x_px, self._WORK_AREA, self._RENDER_PX, image_origin="top_left"
        )
        assert x_mm.shape == (1, 2)
        assert x_mm[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        # Y=0px with top_left -> Y = height_mm (top of image = max mm)
        assert x_mm[0, 1].item() == pytest.approx(297.0, abs=0.5)

    def test_mm_to_px_zero_roundtrip(self):
        """Round-trip mm -> px -> mm should preserve zero coordinates."""
        x_mm = torch.tensor([[0.0, 0.0]])
        x_px = compute.mm_to_px(
            x_mm, self._WORK_AREA, self._RENDER_PX, image_origin="bottom_left"
        )
        x_mm_back = compute.px_to_mm(
            x_px, self._WORK_AREA, self._RENDER_PX, image_origin="bottom_left"
        )
        assert torch.allclose(x_mm, x_mm_back, atol=1e-5)

    def test_mm_to_px_max_boundary(self):
        """Max mm coordinate should map to max pixel coordinate."""
        x_mm = torch.tensor([[210.0, 297.0]])
        x_px = compute.mm_to_px(
            x_mm, self._WORK_AREA, self._RENDER_PX, image_origin="bottom_left"
        )
        assert x_px[0, 0].item() == pytest.approx(908.0, abs=1e-3)
        assert x_px[0, 1].item() == pytest.approx(1280.0, abs=1e-3)

    def test_mm_to_px_center(self):
        """Center mm coordinate should map to center pixel."""
        x_mm = torch.tensor([[105.0, 148.5]])
        x_px = compute.mm_to_px(
            x_mm, self._WORK_AREA, self._RENDER_PX, image_origin="bottom_left"
        )
        assert x_px[0, 0].item() == pytest.approx(454.0, abs=1.0)
        assert x_px[0, 1].item() == pytest.approx(640.0, abs=1.0)

    def test_mm_to_px_invalid_origin(self):
        """Invalid image_origin must raise ValueError."""
        x_mm = torch.tensor([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown image_origin"):
            compute.mm_to_px(
                x_mm, self._WORK_AREA, self._RENDER_PX,
                image_origin="center"
            )

    def test_px_to_mm_invalid_origin(self):
        """Invalid image_origin must raise ValueError."""
        x_px = torch.tensor([[0.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown image_origin"):
            compute.px_to_mm(
                x_px, self._WORK_AREA, self._RENDER_PX,
                image_origin="center"
            )

    def test_mm_to_px_batch(self):
        """Batch of coordinates including zero should all convert correctly."""
        x_mm = torch.tensor([
            [0.0, 0.0],
            [105.0, 148.5],
            [210.0, 297.0],
        ])
        x_px = compute.mm_to_px(
            x_mm, self._WORK_AREA, self._RENDER_PX, image_origin="bottom_left"
        )
        assert x_px.shape == (3, 2)
        # First point: origin
        assert x_px[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert x_px[0, 1].item() == pytest.approx(0.0, abs=1e-5)


# ============================================================================
# ATOMIC YAML DUMP VALIDITY
# ============================================================================

class TestAtomicYamlDump:
    """Test atomic_yaml_dump writes valid, parseable YAML."""

    def test_writes_valid_yaml(self, tmp_path):
        """Output file must be parseable by stdlib yaml."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = tmp_path / "test.yaml"
        fs.atomic_yaml_dump(data, path)

        raw = path.read_text()
        parsed = yaml.safe_load(raw)
        assert parsed == data

    def test_preserves_float_precision(self, tmp_path):
        """Floats should round-trip with reasonable precision."""
        data = {"pi": 3.14159265358979, "small": 1e-7}
        path = tmp_path / "floats.yaml"
        fs.atomic_yaml_dump(data, path)
        loaded = fs.load_yaml(path)
        assert abs(loaded["pi"] - 3.14159265358979) < 1e-10
        assert abs(loaded["small"] - 1e-7) < 1e-12

    def test_handles_empty_dict(self, tmp_path):
        """Empty dict should produce valid YAML."""
        path = tmp_path / "empty.yaml"
        fs.atomic_yaml_dump({}, path)
        loaded = fs.load_yaml(path)
        assert loaded == {} or loaded is None  # YAML may produce None for {}

    def test_handles_nested_lists(self, tmp_path):
        """Nested lists should round-trip."""
        data = {"points": [[1.0, 2.0], [3.0, 4.0]], "tags": ["a", "b"]}
        path = tmp_path / "lists.yaml"
        fs.atomic_yaml_dump(data, path)
        loaded = fs.load_yaml(path)
        assert loaded["points"] == [[1.0, 2.0], [3.0, 4.0]]
        assert loaded["tags"] == ["a", "b"]

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        path = tmp_path / "deep" / "nested" / "config.yaml"
        fs.atomic_yaml_dump({"test": True}, path)
        assert path.exists()
        loaded = fs.load_yaml(path)
        assert loaded["test"] is True

    def test_overwrites_existing_file(self, tmp_path):
        """Should overwrite existing file atomically."""
        path = tmp_path / "overwrite.yaml"
        fs.atomic_yaml_dump({"version": 1}, path)
        fs.atomic_yaml_dump({"version": 2}, path)
        loaded = fs.load_yaml(path)
        assert loaded["version"] == 2

    def test_unicode_content(self, tmp_path):
        """Unicode strings should be preserved."""
        data = {"name": "Bézier", "symbol": "µm"}
        path = tmp_path / "unicode.yaml"
        fs.atomic_yaml_dump(data, path)
        loaded = fs.load_yaml(path)
        assert loaded["name"] == "Bézier"
        assert loaded["symbol"] == "µm"


# ============================================================================
# GEOMETRY BOUNDARY AND DEGENERATE CASES
# ============================================================================

class TestGeometryEdgeCases:
    """Test geometry functions with degenerate and boundary inputs."""

    def test_bezier_degenerate_line(self):
        """Bezier with collinear points should produce a straight line."""
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([10.0, 0.0])
        p3 = torch.tensor([20.0, 0.0])
        p4 = torch.tensor([30.0, 0.0])
        pts = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_mm=0.1)
        # All Y coordinates should be 0
        assert torch.allclose(pts[:, 1], torch.zeros(pts.shape[0]), atol=1e-5)
        # Length should be 30 mm
        length = geometry.polyline_length(pts)
        assert abs(length.item() - 30.0) < 0.1

    def test_bezier_degenerate_point(self):
        """Bezier with all identical points should have zero length."""
        p = torch.tensor([50.0, 50.0])
        pts = geometry.bezier_cubic_polyline(p, p, p, p, max_err_mm=0.5)
        length = geometry.polyline_length(pts)
        assert length.item() < 1e-5

    def test_bezier_eval_at_half(self):
        """Symmetric bezier evaluated at t=0.5 should be at midpoint."""
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([0.0, 100.0])
        p3 = torch.tensor([100.0, 100.0])
        p4 = torch.tensor([100.0, 0.0])
        mid = geometry.bezier_cubic_eval(p1, p2, p3, p4, torch.tensor(0.5))
        # For this symmetric S-curve, midpoint should be at (50, 75)
        assert abs(mid[0].item() - 50.0) < 1.0

    def test_polyline_length_single_point(self):
        """Single-point polyline should have zero length."""
        pts = torch.tensor([[10.0, 20.0]])
        length = geometry.polyline_length(pts)
        assert length.item() == 0.0

    def test_polyline_length_two_points(self):
        """Two-point polyline length should be Euclidean distance."""
        pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        length = geometry.polyline_length(pts)
        assert abs(length.item() - 5.0) < 1e-5

    def test_clip_polyline_fully_inside(self):
        """Polyline fully inside rect should be unchanged."""
        pts = torch.tensor([[20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
        rect = (0.0, 0.0, 100.0, 100.0)
        clipped = geometry.clip_polyline_to_rect(pts, rect)
        assert clipped.shape == pts.shape
        assert torch.allclose(clipped, pts, atol=1e-5)

    def test_clip_polyline_fully_outside(self):
        """Polyline fully outside rect should be empty."""
        pts = torch.tensor([[200.0, 200.0], [300.0, 300.0]])
        rect = (0.0, 0.0, 100.0, 100.0)
        clipped = geometry.clip_polyline_to_rect(pts, rect)
        assert clipped.shape[0] == 0

    def test_parametrize_by_arclength_two_points(self):
        """Arc-length parametrization with two points: [0, 1]."""
        pts = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        s = geometry.parametrize_by_arclength(pts)
        assert s.shape == (2,)
        assert s[0].item() == 0.0
        assert abs(s[1].item() - 1.0) < 1e-5

    def test_interpolate_along_polyline_constant(self):
        """Constant start/end values should produce constant interpolation."""
        pts = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        val = torch.tensor([7.0])
        interp = geometry.interpolate_along_polyline(pts, val, val)
        assert torch.allclose(interp, torch.full((3, 1), 7.0), atol=1e-5)

    def test_bezier_fine_tolerance(self):
        """Very fine tolerance should produce many subdivisions."""
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([50.0, 100.0])
        p3 = torch.tensor([100.0, -50.0])
        p4 = torch.tensor([150.0, 50.0])
        pts_coarse = geometry.bezier_cubic_polyline(
            p1, p2, p3, p4, max_err_mm=5.0
        )
        pts_fine = geometry.bezier_cubic_polyline(
            p1, p2, p3, p4, max_err_mm=0.1
        )
        assert pts_fine.shape[0] > pts_coarse.shape[0]


# ============================================================================
# COMPUTE MODULE EDGE CASES
# ============================================================================

class TestComputeEdgeCases:
    """Edge cases for compute module functions."""

    def test_project_to_bounds_same_bounds(self):
        """When lo == hi, output should equal that value."""
        x = torch.tensor([0.5])
        lo = torch.tensor([1.0])
        hi = torch.tensor([1.0])
        y = compute.project_to_bounds(x, lo, hi)
        assert y.item() == 1.0

    def test_assert_finite_accepts_valid(self):
        """assert_finite should not raise on valid tensor."""
        z = torch.tensor([1.0, 2.0, 3.0])
        # Should not raise
        compute.assert_finite(z, "z")

    def test_normalize_lpips_clamps_above_one(self):
        """Values > 1.0 should be clamped to 1.0 -> mapped to [-1, 1]."""
        img = torch.ones(3, 10, 10) * 1.5
        normalized = compute.normalize_img_for_lpips(img)
        assert torch.all(normalized <= 1.0)
        assert torch.all(normalized >= -1.0)

    def test_normalize_lpips_dtype_fp32(self):
        """Output must always be float32 regardless of input dtype."""
        img = torch.rand(3, 10, 10).half()
        normalized = compute.normalize_img_for_lpips(img)
        assert normalized.dtype == torch.float32

    def test_hann2d_asymmetric(self):
        """Hann window with different H and W."""
        window = compute.hann2d(32, 64)
        assert window.shape == (32, 64)
        assert window[16, 32] > 0.9  # Center peak

    def test_tile_slices_single_tile(self):
        """Image smaller than tile should produce a single tile."""
        slices = compute.tile_slices(H=32, W=32, tile=128, overlap=0)
        assert len(slices) == 1

    def test_clamp_finite_all_valid(self):
        """clamp_finite on all-valid tensor should be identity."""
        x = torch.tensor([0.2, 0.5, 0.8])
        clamped = compute.clamp_finite(x, min_val=0.0, max_val=1.0)
        assert torch.allclose(x, clamped)

    def test_image_mm_to_machine_mm_batch(self):
        """Batch coordinate transform should handle multiple points."""
        pts = torch.tensor([
            [0.0, 0.0],
            [105.0, 148.5],
            [210.0, 297.0],
        ])
        work_area = (210.0, 297.0)
        result = compute.image_mm_to_machine_mm(pts, work_area, flip_y=True)
        assert result.shape == (3, 2)
        # X should be unchanged
        assert torch.allclose(result[:, 0], pts[:, 0])
        # Y should be flipped: y_machine = height - y_image
        assert result[0, 1].item() == pytest.approx(297.0, abs=1e-3)
        assert result[2, 1].item() == pytest.approx(0.0, abs=1e-3)


# ============================================================================
# FS MODULE EDGE CASES
# ============================================================================

class TestFsEdgeCases:
    """Edge cases for filesystem operations."""

    def test_ensure_dir_returns_path(self, tmp_path):
        """ensure_dir should return a Path object."""
        result = fs.ensure_dir(tmp_path / "new_dir")
        assert isinstance(result, Path)
        assert result.exists()

    def test_ensure_dir_accepts_string(self, tmp_path):
        """ensure_dir should accept string paths."""
        str_path = str(tmp_path / "str_dir")
        result = fs.ensure_dir(str_path)
        assert isinstance(result, Path)
        assert result.exists()

    def test_symlink_atomic_to_file(self, tmp_path):
        """Atomic symlink should work for file targets."""
        target = tmp_path / "target.txt"
        target.write_text("hello")
        link = tmp_path / "link"
        fs.symlink_atomic(target, link)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    def test_atomic_save_image_numpy(self, tmp_path):
        """atomic_save_image should accept numpy arrays."""
        import numpy as np
        img = np.random.rand(50, 50, 3).astype(np.float32)
        path = tmp_path / "numpy_img.png"
        fs.atomic_save_image(img, path)
        assert path.exists()


# ============================================================================
# COLOR MODULE EDGE CASES
# ============================================================================

class TestColorEdgeCases:
    """Edge cases for color conversion functions."""

    def test_srgb_to_linear_all_zeros(self):
        """All-zero input should map to all-zero output."""
        zeros = torch.zeros(3, 5, 5)
        linear = color.srgb_to_linear(zeros)
        assert torch.allclose(linear, zeros, atol=1e-7)

    def test_srgb_to_linear_all_ones(self):
        """All-one input should map to all-one output."""
        ones = torch.ones(3, 5, 5)
        linear = color.srgb_to_linear(ones)
        assert torch.allclose(linear, ones, atol=1e-5)

    def test_delta_e2000_black_vs_white(self):
        """DeltaE between black and white should be large."""
        black_lab = torch.zeros(3, 1, 1)  # L=0, a=0, b=0
        white_lab = torch.zeros(3, 1, 1)
        white_lab[0] = 100.0  # L=100
        de = color.delta_e2000(black_lab, white_lab)
        assert de.item() > 50.0  # Should be very large


# ============================================================================
# HASHING EDGE CASES
# ============================================================================

class TestHashingEdgeCases:
    """Edge cases for hashing functions."""

    def test_sha256_tensor_empty(self):
        """Empty tensor should produce a valid hash."""
        t = torch.empty(0)
        h = hashing.sha256_tensor(t)
        assert len(h) == 64

    def test_sha256_tensor_scalar(self):
        """Scalar tensor should produce a valid hash."""
        t = torch.tensor(42.0)
        h = hashing.sha256_tensor(t)
        assert len(h) == 64

    def test_hash_dict_empty(self):
        """Empty dict should produce a valid hash."""
        h = hashing.hash_dict({})
        assert len(h) == 64

    def test_hash_dict_nested_consistency(self):
        """Nested dicts should hash consistently."""
        d = {"a": {"b": {"c": 1}}}
        h1 = hashing.hash_dict(d)
        h2 = hashing.hash_dict(d)
        assert h1 == h2


# ============================================================================
# STROKES EDGE CASES
# ============================================================================

class TestStrokesEdgeCases:
    """Edge cases for strokes module."""

    def test_make_stroke_id_epoch_zero(self):
        """Epoch 0, index 0 should produce a valid ID."""
        sid = strokes.make_stroke_id(epoch=0, idx=0)
        assert isinstance(sid, str)
        assert len(sid) >= 5
        assert "00000" in sid

    def test_stroke_bbox_mm_degenerate(self):
        """Degenerate bezier (all same point) should have zero-area bbox."""
        stroke_dict = {
            "bezier": {
                "p1": [50.0, 50.0],
                "p2": [50.0, 50.0],
                "p3": [50.0, 50.0],
                "p4": [50.0, 50.0],
            }
        }
        xmin, ymin, xmax, ymax = strokes.stroke_bbox_mm(stroke_dict)
        assert xmin == xmax == 50.0
        assert ymin == ymax == 50.0

    def test_extract_stroke_colors_single(self):
        """Single stroke color extraction."""
        strokes_list = [
            {"color_cmy": {"c": 1.0, "m": 0.0, "y": 0.0}},
        ]
        colors = strokes.extract_stroke_colors(strokes_list)
        assert colors.shape == (1, 3)
        assert colors[0, 0].item() == pytest.approx(1.0)

    def test_count_strokes_empty(self, tmp_path):
        """Empty strokes list should count as 0."""
        yaml_file = tmp_path / "empty_strokes.yaml"
        fs.atomic_yaml_dump([], yaml_file)
        count = strokes.count_strokes(yaml_file)
        assert count == 0


# ============================================================================
# INTEGRATION: VALIDATOR + FS ROUND-TRIP
# ============================================================================

@pytest.mark.integration
class TestValidatorFsIntegration:
    """Integration tests spanning validators and fs modules."""

    def test_machine_config_yaml_round_trip(self, tmp_path):
        """Write machine config -> load -> validate -> re-write -> compare."""
        data = {
            "schema": "machine.v1",
            "work_area_mm": {"x": 300.0, "y": 400.0, "z": 50.0},
            "canvas_mm": {
                "x_min": 10.0, "x_max": 290.0,
                "y_min": 10.0, "y_max": 390.0,
            },
            "gcode_flavor": "marlin_2.0",
            "units": "mm",
            "feed_units": "mm/min",
            "feeds": {
                "max_xy_mm_s": 150.0,
                "max_z_mm_s": 25.0,
                "rapid_mm_s": 250.0,
            },
            "acceleration": {
                "max_xy_mm_s2": 800.0,
                "max_z_mm_s2": 400.0,
            },
            "macros": {
                "include_dir": "macros/",
                "purge": "purge.g",
                "pen_up": "pen_up.g",
                "pen_down": "pen_down.g",
            },
            "safety": {
                "soft_limits": True,
                "purge_zone_mm": {
                    "x": [0.0, 10.0],
                    "y": [0.0, 10.0],
                    "z": [0.0, 50.0],
                },
            },
        }
        path1 = tmp_path / "machine1.yaml"
        fs.atomic_yaml_dump(data, path1)
        cfg = validators.load_machine_profile(path1)
        assert cfg.gcode_flavor == "marlin_2.0"
        assert cfg.work_area_mm.x == 300.0

        # Re-serialize and compare
        path2 = tmp_path / "machine2.yaml"
        fs.atomic_yaml_dump(cfg.dict(), path2)
        cfg2 = validators.load_machine_profile(path2)
        assert cfg2.work_area_mm.x == cfg.work_area_mm.x
        assert cfg2.canvas_mm.x_max == cfg.canvas_mm.x_max

    def test_strokes_validate_and_bounds(self, tmp_path):
        """Write strokes -> validate -> extract bounds."""
        stroke_data = {
            "schema": "stroke.v1",
            "strokes": [
                {
                    "id": "test-stroke-00001",
                    "bezier": {
                        "p1": [10.0, 10.0],
                        "p2": [50.0, 50.0],
                        "p3": [100.0, 100.0],
                        "p4": [150.0, 150.0],
                    },
                    "z_profile": {"z0": 5.0, "z1": 15.0},
                    "speed_profile": {"v0": 50.0, "v1": 200.0},
                    "color_cmy": {"c": 0.1, "m": 0.2, "y": 0.3},
                },
            ],
        }
        path = tmp_path / "test_strokes.yaml"
        fs.atomic_yaml_dump(stroke_data, path)
        result = validators.validate_strokes_file(path)
        assert len(result.strokes) == 1

        # Check bbox via strokes module
        raw = fs.load_yaml(path)
        xmin, ymin, xmax, ymax = strokes.stroke_bbox_mm(
            raw["strokes"][0]
        )
        assert xmin >= 10.0
        assert xmax <= 150.0
