#!/usr/bin/env python3
"""Comprehensive test suite for all utils modules - achieving 75%+ coverage.

This combined test suite includes:
- Core functionality tests with hardening improvements
- Seeding for reproducibility
- Tighter normalization checks
- Tiled LPIPS parity testing (even & ragged sizes)
- Logging idempotency and output verification
- Error path testing (bounds projection, assert_finite, broadcasting)
- Geometry quality checks (endpoint preservation, monotonicity)
- Atomic symlink workflow
- Coverage metric on alpha
- File hashing (files & tensors)
- Device recursion
- Coordinate frame transforms (both origins)
- Validators (schema validation, bounds checking)
- G-code generator (coordinate transforms, linearization, soft limits)
- G-code VM (parsing, time estimation, violation detection)
- MLflow helpers (parameter logging, artifact management)
- Integration tests (validators → gcode_gen → vm)
- Edge cases and VM micro-fixes
- Extended coverage for color, metrics, geometry, strokes, torch_utils, hashing

Run with: pytest tests/test_utils_comprehensive.py -v
"""

import json
from pathlib import Path

import pytest
import torch
import numpy as np

from src.utils import (
    color,
    compute,
    fs,
    gcode_generator,
    gcode_vm,
    geometry,
    hashing,
    logging_config,
    metrics,
    mlflow_helpers,
    profiler,
    strokes,
    torch_utils,
    validators,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def machine_cfg(project_root):
    """Load machine configuration (shared across module)."""
    return validators.load_machine_profile(
        project_root / "configs/machine_grbl_airbrush_v1.yaml"
    )


# ============================================================================
# BASIC IMPORT TESTS
# ============================================================================

def test_imports():
    """Test that all utils modules can be imported."""
    assert color is not None
    assert compute is not None
    assert fs is not None
    assert geometry is not None
    assert hashing is not None
    assert logging_config is not None
    assert metrics is not None
    assert profiler is not None
    assert strokes is not None
    assert torch_utils is not None
    assert validators is not None


# ============================================================================
# COLOR TESTS (Target: 98%+ coverage)
# ============================================================================

def test_lpips_normalization():
    """Test LPIPS normalization with tighter checks."""
    torch_utils.seed_everything(123)
    
    img = torch.rand(3, 100, 100)
    normalized = compute.normalize_img_for_lpips(img)
    tol = 1e-6
    
    assert normalized.dtype == torch.float32, "LPIPS input must be float32"
    assert torch.all(normalized >= -1 - tol) and torch.all(normalized <= 1 + tol)


def test_color_roundtrip():
    """Test sRGB ↔ linear color space conversion."""
    torch_utils.seed_everything(123)
    
    srgb = torch.rand(3, 10, 10)
    linear = color.srgb_to_linear(srgb)
    srgb_back = color.linear_to_srgb(linear)
    
    assert torch.allclose(srgb, srgb_back, atol=1e-5)


def test_srgb_to_linear_linear_region():
    """Test sRGB to linear in linear region (small values)."""
    small_vals = torch.tensor([[[0.01, 0.02], [0.03, 0.04]]] * 3)
    linear = color.srgb_to_linear(small_vals)
    # Linear region: x / 12.92
    expected = small_vals / 12.92
    assert torch.allclose(linear, expected, atol=1e-6)


def test_srgb_to_linear_power_region():
    """Test sRGB to linear in power region (large values)."""
    large_vals = torch.ones(3, 5, 5) * 0.5
    linear = color.srgb_to_linear(large_vals)
    # Should use power formula
    assert not torch.allclose(linear, large_vals / 12.92)


def test_linear_to_srgb_linear_region():
    """Test linear to sRGB in linear region."""
    small_vals = torch.tensor([[[0.0001, 0.0002], [0.0003, 0.0004]]] * 3)
    srgb = color.linear_to_srgb(small_vals)
    # Linear region: x * 12.92
    expected = small_vals * 12.92
    assert torch.allclose(srgb, expected, atol=1e-6)


def test_srgb_linear_roundtrip_edge_cases():
    """Test sRGB ↔ linear roundtrip at boundary values."""
    edge = torch.tensor([[[0.04045, 0.0031308], [0.0, 1.0]]] * 3)
    linear = color.srgb_to_linear(edge)
    srgb_back = color.linear_to_srgb(linear)
    assert torch.allclose(edge, srgb_back, atol=1e-5)


def test_luminance_linear_3d():
    """Test luminance calculation for 3D input."""
    img = torch.ones(3, 10, 10)
    lum = color.luminance_linear(img)
    assert lum.shape == (10, 10)
    # White should have luminance ~1
    assert torch.allclose(lum, torch.ones(10, 10), atol=0.01)


def test_luminance_linear_4d():
    """Test luminance calculation for 4D (batched) input."""
    img = torch.rand(3, 3, 8, 8)
    lum = color.luminance_linear(img)
    assert lum.shape == (3, 8, 8)


def test_luminance_linear_invalid_shape():
    """Test luminance raises on invalid shape."""
    img = torch.rand(10, 10)
    with pytest.raises(ValueError, match="Expected shape"):
        color.luminance_linear(img)


def test_rgb_to_xyz():
    """Test RGB to XYZ conversion."""
    white = torch.ones(3, 10, 10)
    xyz = color.rgb_to_xyz(white)
    assert xyz.shape == (3, 10, 10)
    # Check white point Y~1.0
    assert torch.allclose(xyz[1].mean(), torch.tensor(1.0), atol=0.01)


def test_rgb_to_xyz_batched():
    """Test RGB to XYZ with batch dimension."""
    rgb_batch = torch.rand(2, 3, 10, 10)
    xyz_batch = color.rgb_to_xyz(rgb_batch)
    assert xyz_batch.shape == (2, 3, 10, 10)


def test_xyz_to_lab_d65():
    """Test XYZ to Lab conversion with D65 white point."""
    xyz = torch.tensor([[[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]]])
    lab = color.xyz_to_lab(xyz, white_point="D65")
    assert lab.shape == (3, 2, 2)
    # L should be in [0, 100]
    assert torch.all(lab[0] >= -1) and torch.all(lab[0] <= 101)  # Small tolerance


def test_xyz_to_lab_d50():
    """Test XYZ to Lab conversion with D50 white point."""
    xyz = torch.ones(3, 5, 5) * 0.5
    lab = color.xyz_to_lab(xyz, white_point="D50")
    assert lab.shape == (3, 5, 5)


def test_xyz_to_lab_invalid_white_point():
    """Test XYZ to Lab with invalid white point."""
    xyz = torch.ones(3, 5, 5)
    with pytest.raises(ValueError, match="Unknown white_point"):
        color.xyz_to_lab(xyz, white_point="D75")


def test_rgb_to_lab():
    """Test RGB to Lab composite function."""
    rgb = torch.rand(3, 10, 10)
    lab = color.rgb_to_lab(rgb)
    assert lab.shape == (3, 10, 10)
    # Check L range (with tolerance for numerical precision)
    assert torch.all(lab[0] >= -1) and torch.all(lab[0] <= 101)


def test_rgb_to_lab_batched():
    """Test RGB to Lab with batch."""
    rgb = torch.rand(4, 3, 8, 8)
    lab = color.rgb_to_lab(rgb)
    assert lab.shape == (4, 3, 8, 8)


def test_delta_e2000_same_color():
    """Test ΔE2000 for identical colors (should be ~0)."""
    lab = torch.rand(3, 20, 20) * 50 + 25
    de = color.delta_e2000(lab, lab)
    assert de.shape == (20, 20)
    assert torch.allclose(de, torch.zeros_like(de), atol=1e-5)


def test_delta_e2000_different_colors():
    """Test ΔE2000 for different colors (should be > 0)."""
    lab1 = torch.zeros(3, 10, 10)
    lab1[0] = 50.0  # L=50
    
    lab2 = torch.zeros(3, 10, 10)
    lab2[0] = 60.0  # L=60
    
    de = color.delta_e2000(lab1, lab2)
    assert torch.all(de > 0)


def test_delta_e2000_batched():
    """Test ΔE2000 with batch dimension."""
    lab1 = torch.rand(2, 3, 10, 10) * 50
    lab2 = torch.rand(2, 3, 10, 10) * 50
    de = color.delta_e2000(lab1, lab2)
    assert de.shape == (2, 10, 10)


def test_delta_e2000_custom_weights():
    """Test ΔE2000 with custom weighting factors."""
    lab1 = torch.rand(3, 5, 5) * 50
    lab2 = torch.rand(3, 5, 5) * 50
    de = color.delta_e2000(lab1, lab2, kL=2.0, kC=1.5, kH=1.0)
    assert de.shape == (5, 5)


def test_rgb_to_lab_delta_e_workflow():
    """Test complete RGB → Lab → ΔE workflow."""
    rgb1 = torch.rand(3, 32, 32)
    rgb2 = rgb1 + 0.05
    rgb2 = torch.clamp(rgb2, 0, 1)
    
    lab1 = color.rgb_to_lab(rgb1)
    lab2 = color.rgb_to_lab(rgb2)
    de = color.delta_e2000(lab1, lab2)
    
    # Should be small but non-zero
    assert torch.all(de >= 0)


# ============================================================================
# GEOMETRY TESTS (Target: 75%+ coverage)
# ============================================================================

def test_geometry_bezier_monotonicity():
    """Test Bézier subdivision monotonicity and endpoint preservation."""
    torch_utils.seed_everything(123)
    
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([50.0, 0.0])
    p3 = torch.tensor([100.0, 50.0])
    p4 = torch.tensor([150.0, 50.0])
    
    pts1 = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_mm=1.0)
    pts2 = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_mm=0.5)
    pts3 = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_mm=0.25)
    
    assert pts1.shape[0] <= pts2.shape[0] <= pts3.shape[0]
    
    len1 = geometry.polyline_length(pts1)
    len3 = geometry.polyline_length(pts3)
    assert len3 >= len1 and (len3 - len1) / (len3 + 1e-9) < 0.02
    
    assert torch.allclose(pts1[0], p1, atol=1e-5)
    assert torch.allclose(pts1[-1], p4, atol=1e-5)


def test_bezier_cubic_eval():
    """Test cubic Bézier evaluation at specific t."""
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0])
    p3 = torch.tensor([1.0, 1.0])
    p4 = torch.tensor([2.0, 1.0])
    
    # Evaluate at t=0 (should be p1)
    pt = geometry.bezier_cubic_eval(p1, p2, p3, p4, torch.tensor(0.0))
    assert torch.allclose(pt, p1, atol=1e-5)
    
    # Evaluate at t=1 (should be p4)
    pt = geometry.bezier_cubic_eval(p1, p2, p3, p4, torch.tensor(1.0))
    assert torch.allclose(pt, p4, atol=1e-5)


def test_bezier_cubic_eval_multiple_t():
    """Test Bézier evaluation at multiple t values."""
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0])
    p3 = torch.tensor([1.0, 1.0])
    p4 = torch.tensor([2.0, 1.0])
    
    # Evaluate at multiple t values separately (function doesn't support batched t with 1D points)
    t_vals = [0.0, 0.5, 1.0]
    pts = torch.stack([
        geometry.bezier_cubic_eval(p1, p2, p3, p4, torch.tensor(t))
        for t in t_vals
    ])
    
    assert pts.shape == (3, 2)  # 3 t values, 2D points
    # Verify endpoints
    assert torch.allclose(pts[0], p1, atol=1e-5)
    assert torch.allclose(pts[-1], p4, atol=1e-5)
    # Middle point should be between start and end
    assert torch.all(pts[1] >= torch.min(p1, p4))
    assert torch.all(pts[1] <= torch.max(p1, p4))


def test_polyline_bbox():
    """Test polyline bounding box."""
    pts = torch.tensor([[10.0, 20.0], [50.0, 60.0], [30.0, 40.0]])
    xmin, ymin, xmax, ymax = geometry.polyline_bbox(pts)
    assert xmin == 10.0
    assert xmax == 50.0
    assert ymin == 20.0
    assert ymax == 60.0


def test_polyline_bbox_single_point():
    """Test bbox with single point."""
    pts = torch.tensor([[25.0, 35.0]])
    xmin, ymin, xmax, ymax = geometry.polyline_bbox(pts)
    assert xmin == xmax == 25.0
    assert ymin == ymax == 35.0


def test_polyline_bbox_empty():
    """Test bbox with empty polyline."""
    pts = torch.empty(0, 2)
    bbox = geometry.polyline_bbox(pts)
    assert bbox == (0.0, 0.0, 0.0, 0.0)


def test_clip_polyline_to_rect():
    """Test polyline clipping to rectangle."""
    pts = torch.tensor([[0.0, 0.0], [50.0, 50.0], [150.0, 150.0]])
    rect = (10.0, 10.0, 100.0, 100.0)
    clipped = geometry.clip_polyline_to_rect(pts, rect)
    assert clipped.shape[0] >= 1
    assert torch.all(clipped[:, 0] >= 10.0) and torch.all(clipped[:, 0] <= 100.0)
    assert torch.all(clipped[:, 1] >= 10.0) and torch.all(clipped[:, 1] <= 100.0)


def test_parametrize_by_arclength():
    """Test arc-length parametrization."""
    pts = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    s = geometry.parametrize_by_arclength(pts)
    assert s.shape == (3,)
    assert s[0] == 0.0
    assert s[-1] == 1.0


def test_stroke_bbox_mm():
    """Test stroke bounding box calculation."""
    p1 = torch.tensor([10.0, 20.0])
    p2 = torch.tensor([30.0, 40.0])
    p3 = torch.tensor([50.0, 60.0])
    p4 = torch.tensor([70.0, 80.0])
    
    pts = geometry.bezier_cubic_polyline(p1, p2, p3, p4, max_err_mm=1.0)
    xmin, ymin, xmax, ymax = geometry.polyline_bbox(pts)
    
    assert xmin >= 10.0
    assert xmax <= 70.0
    assert ymin >= 20.0
    assert ymax <= 80.0


def test_interpolate_along_polyline():
    """Test interpolation along polyline."""
    pts = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    # Function interpolates scalar values (like Z or speed) along the polyline
    # Not the polyline points themselves
    values_start = torch.tensor([5.0])  # e.g., Z at start
    values_end = torch.tensor([15.0])   # e.g., Z at end
    
    interp_values = geometry.interpolate_along_polyline(pts, values_start, values_end)
    # Returns (N, D) where N=num points, D=dimension of values
    assert interp_values.shape == (3, 1)
    # First value should be close to start
    assert torch.allclose(interp_values[0], values_start, atol=1e-5)
    # Last value should be close to end
    assert torch.allclose(interp_values[-1], values_end, atol=1e-5)


# ============================================================================
# STROKES TESTS (Target: 75%+ coverage)
# ============================================================================

def test_strokes_vec_yaml_roundtrip():
    """Test stroke vector ↔ YAML roundtrip with bounds validation."""
    torch_utils.seed_everything(123)
    
    vec_mm = torch.randn(15)
    bounds = {
        "x": (0.0, 210.0), "y": (0.0, 297.0),
        "z": (0.0, 30.0), "speed": (1.0, 300.0), "cmy": (0.0, 1.0)
    }
    
    stroke_dict = strokes.stroke_vec_to_yaml_dict(vec_mm, bounds=bounds)
    vec_back = strokes.stroke_yaml_dict_to_vec(stroke_dict)
    assert vec_mm.shape == vec_back.shape
    
    _ = validators.StrokeV1(**stroke_dict)


def test_make_stroke_id():
    """Test stroke ID generation."""
    sid = strokes.make_stroke_id(epoch=5, idx=42)
    assert isinstance(sid, str)
    # Format is "EEEEE-IIIII-HHHHHHHH" (epoch-index-hash)
    # Should contain 00005 (epoch) and 00042 (index)
    assert '00005' in sid
    assert '00042' in sid
    # Should have two dashes separating the parts
    assert sid.count('-') == 2


def test_stroke_bbox_mm():
    """Test stroke bounding box from YAML."""
    stroke_dict = {
        'bezier': {
            'p1': [10.0, 20.0],
            'p2': [30.0, 40.0],
            'p3': [50.0, 60.0],
            'p4': [70.0, 80.0],
        }
    }
    xmin, ymin, xmax, ymax = strokes.stroke_bbox_mm(stroke_dict)
    assert xmin >= 10.0
    assert xmax <= 70.0


def test_count_strokes(tmp_path):
    """Test counting strokes in YAML file."""
    # count_strokes expects a list directly, not a dict with 'strokes' key
    strokes_data = [
        {'id': f'stroke-{i}', 'bezier': {'p1': [0,0], 'p2': [1,1], 'p3': [2,2], 'p4': [3,3]},
         'z_profile': {'z0': 10, 'z1': 10}, 'speed_profile': {'v0': 100, 'v1': 100},
         'color_cmy': {'c': 0, 'm': 0, 'y': 0}}
        for i in range(5)
    ]
    
    yaml_file = tmp_path / "strokes.yaml"
    fs.atomic_yaml_dump(strokes_data, yaml_file)
    
    count = strokes.count_strokes(yaml_file)
    assert count == 5


def test_extract_stroke_colors():
    """Test extracting colors from strokes."""
    strokes_list = [
        {'color_cmy': {'c': 0.1, 'm': 0.2, 'y': 0.3}},
        {'color_cmy': {'c': 0.4, 'm': 0.5, 'y': 0.6}},
    ]
    
    colors = strokes.extract_stroke_colors(strokes_list)
    assert colors.shape == (2, 3)
    assert torch.allclose(colors[0], torch.tensor([0.1, 0.2, 0.3]))


# ============================================================================
# METRICS TESTS (Target: 75%+ coverage)
# ============================================================================

def test_psnr_identical_images():
    """Test PSNR for identical images (should be very high)."""
    img = torch.rand(3, 32, 32)
    psnr_val = metrics.psnr(img, img)
    assert psnr_val.item() > 70  # Very high PSNR


def test_psnr_different_images():
    """Test PSNR for different images."""
    img1 = torch.zeros(3, 32, 32)
    img2 = torch.ones(3, 32, 32) * 0.5
    psnr_val = metrics.psnr(img1, img2)
    assert 0 < psnr_val.item() < 100


def test_psnr_batched():
    """Test PSNR with batch dimension (returns scalar mean)."""
    img1 = torch.rand(4, 3, 32, 32)
    img2 = torch.rand(4, 3, 32, 32)
    psnr_val = metrics.psnr(img1, img2)
    # Returns scalar (mean over batch)
    assert psnr_val.shape == torch.Size([])


def test_ssim_identical_images():
    """Test SSIM for identical images (should be 1.0)."""
    img = torch.rand(1, 3, 64, 64)
    ssim_val = metrics.ssim(img, img)
    assert torch.allclose(ssim_val, torch.tensor(1.0), atol=1e-3)


def test_ssim_different_images():
    """Test SSIM for different images."""
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    ssim_val = metrics.ssim(img1, img2)
    assert 0 < ssim_val.item() < 1.0


def test_ssim_window_size():
    """Test SSIM with custom window size."""
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    ssim_val = metrics.ssim(img1, img2, window_size=7)
    assert 0 < ssim_val.item() <= 1.0


def test_edge_preservation_score():
    """Test edge preservation score."""
    target = torch.rand(3, 64, 64)
    pred = torch.rand(3, 64, 64)
    score = metrics.edge_preservation_score(target, pred)
    assert 0 <= score.item() <= 1.0


def test_edge_preservation_perfect():
    """Test edge preservation with identical images."""
    img = torch.rand(3, 64, 64)
    score = metrics.edge_preservation_score(img, img)
    assert torch.allclose(score, torch.tensor(1.0), atol=0.1)


def test_paint_coverage_painted():
    """Test coverage metric on fully painted canvas."""
    canvas_black = torch.zeros(64, 64)
    cov_black = metrics.paint_coverage(canvas_black)
    assert abs(cov_black.item() - 1.0) < 1e-6


def test_paint_coverage_unpainted():
    """Test coverage metric on unpainted canvas."""
    canvas_white = torch.ones(64, 64)
    cov_white = metrics.paint_coverage(canvas_white)
    assert abs(cov_white.item() - 0.0) < 1e-6


# ============================================================================
# TORCH_UTILS TESTS (Target: 75%+ coverage)
# ============================================================================

def test_seed_everything_reproducibility():
    """Test seed_everything produces reproducible results."""
    torch_utils.seed_everything(42)
    x1 = torch.rand(5, 5)
    
    torch_utils.seed_everything(42)
    x2 = torch.rand(5, 5)
    
    assert torch.allclose(x1, x2)


def test_to_device_recursive():
    """Test recursive device movement."""
    t = torch.randn(2, 3)
    nested = {
        "a": t.clone(),
        "b": [t.clone(), {"c": t.clone()}],
        "scalar": 42
    }
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = torch_utils.to_device_recursive(nested, dev)
    
    assert moved["a"].device == dev
    assert moved["b"][0].device == dev
    assert moved["b"][1]["c"].device == dev
    assert moved["scalar"] == 42


def test_set_channels_last():
    """Test setting channels-last memory format."""
    model = torch.nn.Conv2d(3, 64, 3)
    torch_utils.set_channels_last(model)
    # Check first conv layer is channels last
    for param in model.parameters():
        if param.ndim == 4:
            assert param.is_contiguous(memory_format=torch.channels_last)
            break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_gpu_mem_highwater():
    """Test GPU memory high-water mark tracking."""
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    
    _ = torch.rand(1000, 1000, device=device)
    
    hwm_bytes = torch_utils.get_gpu_mem_highwater(reset=False)
    assert hwm_bytes > 0


def test_count_parameters():
    """Test counting model parameters."""
    model = torch.nn.Linear(10, 5)
    total = torch_utils.count_parameters(model, trainable_only=False)
    assert total == 55  # 10*5 + 5 bias


def test_count_parameters_trainable_only():
    """Test counting only trainable parameters."""
    model = torch.nn.Linear(10, 5)
    model.weight.requires_grad = False
    trainable = torch_utils.count_parameters(model, trainable_only=True)
    assert trainable == 5  # Only bias


def test_get_model_size():
    """Test getting model size information."""
    model = torch.nn.Linear(10, 5)
    size_info = torch_utils.get_model_size(model)
    assert 'total_params' in size_info
    assert 'trainable_params' in size_info
    assert size_info['total_params'] == 55


# ============================================================================
# COMPUTE TESTS (Target: 75%+ coverage)
# ============================================================================

def test_mm_px_roundtrip():
    """Test mm ↔ px coordinate conversion roundtrip."""
    torch_utils.seed_everything(123)
    
    x_mm = torch.tensor([[50.0, 100.0], [100.0, 200.0]])
    work_area = (210.0, 297.0)
    render_px = (908, 1280)
    
    x_px = compute.mm_to_px(x_mm, work_area, render_px)
    x_mm_back = compute.px_to_mm(x_px, work_area, render_px)
    
    assert torch.allclose(x_mm, x_mm_back, atol=0.01)


def test_mm_px_both_origins():
    """Test mm ↔ px conversion with both image origins."""
    torch_utils.seed_everything(123)
    
    x_mm = torch.tensor([[50.0, 100.0], [100.0, 200.0]])
    work_area = (210.0, 297.0)
    render_px = (908, 1280)
    
    x_px_top = compute.mm_to_px(x_mm, work_area, render_px, image_origin="top_left")
    x_px_bottom = compute.mm_to_px(x_mm, work_area, render_px, image_origin="bottom_left")
    
    assert not torch.allclose(x_px_top, x_px_bottom)


def test_normalize_img_for_lpips_out_of_range():
    """Test LPIPS normalization handles out-of-range values."""
    img = torch.rand(3, 32, 32) * 2.0
    normalized = compute.normalize_img_for_lpips(img)
    assert torch.all(normalized >= -1.0) and torch.all(normalized <= 1.0)


def test_hann2d_shapes():
    """Test 2D Hann window generation."""
    window = compute.hann2d(64, 64)
    assert window.shape == (64, 64)
    # Center should be near 1, edges near 0
    assert window[32, 32] > 0.9
    assert window[0, 0] < 0.1


def test_tile_slices():
    """Test tile slice generation."""
    slices = compute.tile_slices(H=256, W=256, tile=128, overlap=32)
    # With overlap, we get more tiles than simple division
    # 256 / (128 - 32) = 2.67, so we need 3 tiles per dimension
    # Total: 3x3 = 9 tiles
    assert len(slices) == 9
    # Verify all slices are within bounds
    for (h_slice, w_slice) in slices:
        assert h_slice.start >= 0
        assert w_slice.start >= 0
        assert h_slice.stop <= 256
        assert w_slice.stop <= 256


def test_tile_slices_no_overlap():
    """Test tile slices without overlap."""
    slices = compute.tile_slices(H=128, W=128, tile=64, overlap=0)
    assert len(slices) == 4


def test_project_to_bounds_correctness():
    """Test bounds projection correctness."""
    x = torch.tensor([-10.0, 0.5, 999.0])
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([1.0, 1.0, 10.0])
    y = compute.project_to_bounds(x, lo, hi)
    
    expected = torch.tensor([0.0, 0.5, 10.0])
    assert torch.allclose(y, expected)


def test_project_to_bounds_idempotence():
    """Test bounds projection idempotence."""
    x = torch.tensor([-10.0, 0.5, 999.0])
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([1.0, 1.0, 10.0])
    y = compute.project_to_bounds(x, lo, hi)
    
    y2 = compute.project_to_bounds(y, lo, hi)
    assert torch.equal(y, y2)


def test_project_to_bounds_broadcast():
    """Test bounds projection with broadcasting."""
    x_broadcast = torch.tensor([-1.0, 2.0])
    lo_scalar = torch.tensor(0.0)
    hi_broadcast = torch.tensor([1.0, 1.0])
    y_broadcast = compute.project_to_bounds(x_broadcast, lo_scalar, hi_broadcast)
    assert torch.allclose(y_broadcast, torch.tensor([0.0, 1.0]))


def test_assert_finite_raises_on_nan():
    """Test assert_finite raises on NaN."""
    z = torch.tensor([0.0, float('nan')])
    with pytest.raises(ValueError):
        compute.assert_finite(z, "z")


def test_assert_finite_raises_on_inf():
    """Test assert_finite raises on Inf."""
    z_inf = torch.tensor([0.0, float('inf')])
    with pytest.raises(ValueError):
        compute.assert_finite(z_inf, "z_inf")


def test_clamp_finite():
    """Test clamp_finite replaces NaN/Inf."""
    x = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 0.5])
    # Correct parameter names: min_val and max_val
    clamped = compute.clamp_finite(x, min_val=0.0, max_val=1.0)
    assert not torch.isnan(clamped).any()
    assert not torch.isinf(clamped).any()
    assert torch.all(clamped >= 0.0) and torch.all(clamped <= 1.0)


def test_coordinate_frame_transform():
    """Test coordinate frame transform (image → machine)."""
    W, H = 210.0, 297.0
    
    pts_img = torch.tensor([[0.0, 0.0], [W, H]])
    pts_mach = compute.image_mm_to_machine_mm(pts_img, (W, H), flip_y=True)
    
    assert torch.allclose(pts_mach[0], torch.tensor([0.0, H]))
    assert torch.allclose(pts_mach[1], torch.tensor([W, 0.0]))


# ============================================================================
# HASHING TESTS (Target: 75%+ coverage)
# ============================================================================

def test_sha256_tensor_deterministic_cpu():
    """Test tensor hashing is deterministic on CPU."""
    t = torch.randn(100, 100)
    h1 = hashing.sha256_tensor(t)
    h2 = hashing.sha256_tensor(t)
    assert h1 == h2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sha256_tensor_gpu():
    """Test tensor hashing works with GPU tensors."""
    t = torch.randn(50, 50, device='cuda')
    h = hashing.sha256_tensor(t)
    assert len(h) == 64


def test_sha256_tensor_different_dtypes():
    """Test hashing tensors with different dtypes."""
    t_float = torch.ones(10, 10, dtype=torch.float32)
    t_double = torch.ones(10, 10, dtype=torch.float64)
    h_float = hashing.sha256_tensor(t_float)
    h_double = hashing.sha256_tensor(t_double)
    assert h_float != h_double


def test_sha256_file_large_file(tmp_path):
    """Test hashing a larger file."""
    large_file = tmp_path / "large.bin"
    data = b"x" * (10 * 1024 * 1024)
    large_file.write_bytes(data)
    
    h = hashing.sha256_file(large_file)
    assert len(h) == 64


def test_sha256_file_empty(tmp_path):
    """Test hashing an empty file."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    
    h = hashing.sha256_file(empty_file)
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_dict():
    """Test dictionary hashing."""
    d = {'a': 1, 'b': [2, 3], 'c': {'d': 4}}
    h = hashing.hash_dict(d)
    assert len(h) == 64
    
    h2 = hashing.hash_dict(d)
    assert h == h2


def test_hash_dict_order_independence():
    """Test dict hashing is order-independent."""
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 2, 'a': 1}
    h1 = hashing.hash_dict(d1)
    h2 = hashing.hash_dict(d2)
    assert h1 == h2


def test_file_hashing_detects_changes(tmp_path):
    """Test file hashing detects content changes."""
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    h1 = hashing.sha256_file(p)
    
    p.write_bytes(b"abcd")
    h2 = hashing.sha256_file(p)
    
    assert h1 != h2
    assert len(h1) == 64 and len(h2) == 64


# ============================================================================
# FILESYSTEM (FS) TESTS (Target: 75%+ coverage)
# ============================================================================

def test_ensure_dir_creates_directory(tmp_path):
    """Test ensure_dir creates directory."""
    new_dir = tmp_path / "new" / "nested" / "dir"
    fs.ensure_dir(new_dir)
    assert new_dir.exists()


def test_ensure_dir_idempotent(tmp_path):
    """Test ensure_dir is idempotent."""
    new_dir = tmp_path / "test_dir"
    fs.ensure_dir(new_dir)
    fs.ensure_dir(new_dir)
    assert new_dir.exists()


def test_atomic_image_save(tmp_path):
    """Test atomic image save."""
    test_img = torch.rand(3, 50, 50)
    fs.atomic_save_image(test_img, tmp_path / 'test.png')
    assert (tmp_path / 'test.png').exists()


def test_atomic_image_save_overwrites(tmp_path):
    """Test atomic save overwrites existing file."""
    img_path = tmp_path / "test.png"
    img1 = torch.rand(3, 32, 32)
    img2 = torch.rand(3, 32, 32)
    
    fs.atomic_save_image(img1, img_path)
    fs.atomic_save_image(img2, img_path)
    assert img_path.exists()


def test_atomic_yaml(tmp_path):
    """Test atomic YAML write and load."""
    fs.atomic_yaml_dump({'test': 'data', 'value': 42}, tmp_path / 'test.yaml')
    loaded = fs.load_yaml(tmp_path / 'test.yaml')
    assert loaded['value'] == 42


def test_atomic_yaml_dump_pretty(tmp_path):
    """Test atomic YAML dump with formatting."""
    data = {'nested': {'key': 'value'}, 'list': [1, 2, 3]}
    yaml_file = tmp_path / "pretty.yaml"
    
    fs.atomic_yaml_dump(data, yaml_file)
    
    content = yaml_file.read_text()
    assert 'nested:' in content


def test_atomic_symlink_flip(tmp_path):
    """Test atomic symlink replacement."""
    e1 = tmp_path / "epoch_1"
    e2 = tmp_path / "epoch_2"
    link = tmp_path / "latest"
    e1.mkdir()
    e2.mkdir()
    
    fs.symlink_atomic(e1, link)
    assert link.exists() and link.resolve() == e1.resolve()
    
    fs.symlink_atomic(e2, link)
    assert link.exists() and link.resolve() == e2.resolve()


def test_load_yaml_with_includes(tmp_path):
    """Test YAML loading."""
    yaml_content = """
    name: test
    value: 42
    """
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    
    data = fs.load_yaml(yaml_file)
    assert data['name'] == 'test'
    assert data['value'] == 42


# ============================================================================
# TILED LPIPS TESTS
# ============================================================================

def test_tiled_lpips_parity_even():
    """Test tiled LPIPS matches full-frame for even tile sizes."""
    torch_utils.seed_everything(456)
    
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    img1_norm = compute.normalize_img_for_lpips(img1.squeeze(0)).unsqueeze(0)
    img2_norm = compute.normalize_img_for_lpips(img2.squeeze(0)).unsqueeze(0)
    
    lpips_full = compute.TiledLPIPS(net='alex', tile_size=0)
    d_full = lpips_full(img1_norm, img2_norm).item()
    
    lpips_tiled = compute.TiledLPIPS(net='alex', tile_size=128, overlap=32)
    d_tiled = lpips_tiled(img1_norm, img2_norm).item()
    
    diff = abs(d_full - d_tiled)
    assert diff < 0.1


def test_tiled_lpips_parity_ragged():
    """Test tiled LPIPS matches full-frame for ragged (non-divisible) sizes."""
    torch_utils.seed_everything(456)
    
    img1_ragged = torch.rand(1, 3, 250, 250)
    img2_ragged = torch.rand(1, 3, 250, 250)
    img1_ragged_norm = compute.normalize_img_for_lpips(img1_ragged.squeeze(0)).unsqueeze(0)
    img2_ragged_norm = compute.normalize_img_for_lpips(img2_ragged.squeeze(0)).unsqueeze(0)
    
    lpips_full_ragged = compute.TiledLPIPS(net='alex', tile_size=0)
    d_full_ragged = lpips_full_ragged(img1_ragged_norm, img2_ragged_norm).item()
    
    lpips_tiled_ragged = compute.TiledLPIPS(net='alex', tile_size=128, overlap=32)
    d_tiled_ragged = lpips_tiled_ragged(img1_ragged_norm, img2_ragged_norm).item()
    
    diff_ragged = abs(d_full_ragged - d_tiled_ragged)
    assert diff_ragged < 0.1


# ============================================================================
# LOGGING TESTS
# ============================================================================

def test_logging_idempotency(tmp_path):
    """Test logging file output and idempotency."""
    import contextlib
    import io
    
    log_path = tmp_path / "test.log"
    
    errbuf = io.StringIO()
    with contextlib.redirect_stderr(errbuf):
        logging_config.setup_logging(
            log_level="INFO",
            log_file=str(log_path),
            json=True,
            to_stderr=False,
            context={"app": "test"}
        )
        logger = logging_config.get_logger("utils_test")
        logger.info("hello", extra={"trial": 7})
        
        logging_config.setup_logging(
            log_level="INFO",
            log_file=str(log_path),
            json=True,
            to_stderr=False,
            context={"app": "test"}
        )
        logger.info("world")
    
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    
    rec = json.loads(lines[0])
    assert rec["msg"] == "hello"
    assert rec.get("app") == "test"


# ============================================================================
# PROFILER TESTS
# ============================================================================

def test_profiler_timer():
    """Test profiler timer functionality."""
    times = []
    with profiler.timer('test_op', sink=lambda n, t: times.append(t)):
        _ = torch.rand(500, 500) @ torch.rand(500, 500)
    
    assert len(times) == 1 and times[0] > 0


def test_profiler_nvtx():
    """Test NVTX range (no-op if unavailable)."""
    with profiler.nvtx_range("test_range"):
        pass


# ============================================================================
# VALIDATORS TESTS
# ============================================================================

def test_validators_stroke_bounds(project_root):
    """Test get_stroke_bounds returns correct canvas dimensions."""
    bounds = validators.get_stroke_bounds(
        project_root / "configs/machine_grbl_airbrush_v1.yaml"
    )
    assert 'x' in bounds and 'y' in bounds and 'z' in bounds
    assert bounds['x'] == (20.0, 230.0)
    assert bounds['y'] == (26.5, 323.5)


def test_validators_valid_stroke():
    """Test valid stroke passes validation."""
    stroke_data = {
        'id': 'test-stroke-12345',
        'bezier': {
            'p1': [10.0, 10.0],
            'p2': [50.0, 20.0],
            'p3': [100.0, 30.0],
            'p4': [150.0, 40.0],
        },
        'z_profile': {'z0': 10.0, 'z1': 15.0},
        'speed_profile': {'v0': 100.0, 'v1': 150.0},
        'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
    }
    stroke = validators.StrokeV1(**stroke_data)
    assert stroke.id == 'test-stroke-12345'


def test_validators_out_of_bounds_stroke():
    """Test out-of-bounds stroke is rejected."""
    stroke_data = {
        'id': 'test-stroke-12345',
        'bezier': {
            'p1': [10.0, 10.0],
            'p2': [50.0, 20.0],
            'p3': [100.0, 30.0],
            'p4': [300.0, 40.0],
        },
        'z_profile': {'z0': 10.0, 'z1': 15.0},
        'speed_profile': {'v0': 100.0, 'v1': 150.0},
        'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
    }
    with pytest.raises(ValueError, match='out of bounds'):
        validators.StrokeV1(**stroke_data)


def test_validators_machine_profile(project_root):
    """Test load_machine_profile works with real config."""
    machine_cfg = validators.load_machine_profile(
        project_root / "configs/machine_grbl_airbrush_v1.yaml"
    )
    assert machine_cfg.schema_version == "machine.v1"
    assert machine_cfg.work_area_mm.x == 250.0
    assert machine_cfg.canvas_mm.x_min == 20.0
    assert machine_cfg.canvas_mm.x_max == 230.0
    assert machine_cfg.gcode_flavor == "grbl_1.1f"


def test_validators_flatten_config():
    """Test flatten_config_for_mlflow correctly flattens nested dicts."""
    nested_cfg = {
        'agent': {
            'learning_rate': 0.001,
            'gamma': 0.99,
        },
        'env': {
            'render_px': 908,
        }
    }
    flat = validators.flatten_config_for_mlflow(nested_cfg)
    assert flat['agent.learning_rate'] == 0.001
    assert flat['agent.gamma'] == 0.99
    assert flat['env.render_px'] == 908


# ============================================================================
# G-CODE TESTS
# ============================================================================

def test_gcode_coordinate_transform(machine_cfg):
    """Test G-code coordinate frame transform with canvas offset."""
    pts_img = torch.tensor([
        [0.0, 0.0],
        [210.0, 0.0],
        [0.0, 297.0],
        [210.0, 297.0],
        [105.0, 148.5],
    ])
    
    canvas_bounds = (20.0, 230.0, 26.5, 323.5)
    pts_mach = gcode_generator.image_mm_to_machine_mm(
        pts_img, canvas_bounds, flip_y=True
    )
    
    assert torch.allclose(pts_mach[0], torch.tensor([20.0, 323.5]), atol=1e-6)
    assert torch.allclose(pts_mach[1], torch.tensor([230.0, 323.5]), atol=1e-6)
    assert torch.allclose(pts_mach[2], torch.tensor([20.0, 26.5]), atol=1e-6)
    assert torch.allclose(pts_mach[3], torch.tensor([230.0, 26.5]), atol=1e-6)
    assert torch.allclose(pts_mach[4], torch.tensor([125.0, 175.0]), atol=1e-6)


def test_gcode_header_generation(machine_cfg):
    """Test G-code header contains required commands."""
    header = gcode_generator.generate_gcode_header(machine_cfg)
    header_str = ''.join(header)
    assert 'G21' in header_str
    assert 'G90' in header_str
    assert 'grbl_1.1f' in header_str


def test_gcode_feed_conversion():
    """Test speed → feed conversion."""
    feed_mm_min = gcode_generator.speed_mm_s_to_feed(100.0, "mm/min")
    assert feed_mm_min == 6000.0
    feed_mm_s = gcode_generator.speed_mm_s_to_feed(100.0, "mm/s")
    assert feed_mm_s == 100.0


def test_gcode_soft_limit_validation(machine_cfg):
    """Test soft limit violation is detected."""
    with pytest.raises(ValueError, match='out of bounds'):
        gcode_generator.validate_soft_limits(260.0, 100.0, 10.0, machine_cfg)


def test_gcode_stroke_linearization(machine_cfg):
    """Test stroke linearization produces valid polyline."""
    stroke_data = {
        'id': 'test-stroke-001',
        'bezier': {
            'p1': [50.0, 50.0],
            'p2': [70.0, 60.0],
            'p3': [90.0, 70.0],
            'p4': [110.0, 80.0],
        },
        'z_profile': {'z0': 10.0, 'z1': 15.0},
        'speed_profile': {'v0': 100.0, 'v1': 150.0},
        'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
    }
    stroke = validators.StrokeV1(**stroke_data)
    
    pts_mach, z_interp, v_interp = gcode_generator.linearize_stroke(
        stroke, machine_cfg, max_err_mm=0.5
    )
    
    assert pts_mach.shape[0] >= 2
    assert z_interp.shape[0] == pts_mach.shape[0]
    assert v_interp.shape[0] == pts_mach.shape[0]
    assert torch.all(z_interp >= 10.0) and torch.all(z_interp <= 15.0)
    assert torch.all(v_interp >= 100.0) and torch.all(v_interp <= 150.0)


def test_gcode_full_generation(machine_cfg, tmp_path):
    """Test full G-code generation writes valid file."""
    stroke_data = {
        'id': 'test-stroke-001',
        'bezier': {
            'p1': [50.0, 50.0],
            'p2': [70.0, 60.0],
            'p3': [90.0, 70.0],
            'p4': [110.0, 80.0],
        },
        'z_profile': {'z0': 10.0, 'z1': 15.0},
        'speed_profile': {'v0': 100.0, 'v1': 150.0},
        'color_cmy': {'c': 0.5, 'm': 0.3, 'y': 0.7},
    }
    
    strokes_file_data = {
        'schema': 'stroke.v1',
        'strokes': [stroke_data]
    }
    strokes_obj = validators.StrokesFileV1(**strokes_file_data)
    
    gcode_path = tmp_path / "test.gcode"
    gcode_generator.generate_cmy_gcode(
        strokes_obj, machine_cfg, gcode_path
    )
    
    assert gcode_path.exists()
    gcode_content = gcode_path.read_text()
    assert 'G21' in gcode_content
    assert 'G1' in gcode_content
    assert 'STROKE_ID: test-stroke-001' in gcode_content


# ============================================================================
# G-CODE VM TESTS
# ============================================================================

def test_gcode_vm_simple_execution(machine_cfg):
    """Test VM executes simple G-code without violations."""
    vm = gcode_vm.GCodeVM(machine_cfg, purge_time_s=1.0, pen_time_s=0.2)
    
    test_gcode = """
; Test G-code
G21
G90
G1 X10 Y10 Z5 F3000
G1 X20 Y20 Z5 F3000
G1 X30 Y20 Z5 F3000
"""
    vm.load_string(test_gcode)
    result = vm.run()
    
    assert result['move_count'] == 3
    assert result['time_estimate_s'] > 0
    assert len(result['violations']) == 0
    assert result['final_pos'][0] == 30.0
    assert result['final_pos'][1] == 20.0


def test_gcode_vm_soft_limit_violation(machine_cfg):
    """Test VM detects soft limit violations."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    bad_gcode = """
G21
G90
G1 X260 Y100 Z5 F3000
"""
    vm.load_string(bad_gcode)
    result = vm.run()
    
    assert len(result['violations']) > 0
    assert 'machine limit' in result['violations'][0].lower()


def test_gcode_vm_stroke_id_tracking(machine_cfg):
    """Test VM tracks stroke IDs from comments."""
    gcode_with_id = """
; STROKE_ID: stroke-abc-123
G1 X10 Y10 Z5 F3000
"""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_string(gcode_with_id)
    result = vm.run()
    
    assert result['last_stroke_id'] == 'stroke-abc-123'


def test_gcode_vm_time_estimation(machine_cfg):
    """Test VM time estimation with acceleration."""
    vm = gcode_vm.GCodeVM(machine_cfg, accel_mm_s2=1000.0)
    move_gcode = """
G21
G90
G1 X100 Y0 Z0 F6000
"""
    vm.load_string(move_gcode)
    result = vm.run()
    
    assert 0.5 < result['time_estimate_s'] < 5.0


def test_gcode_vm_trajectory_extraction(machine_cfg):
    """Test VM extracts trajectory correctly."""
    test_gcode = """
; Test G-code
G21
G90
G1 X10 Y10 Z5 F3000
G1 X20 Y20 Z5 F3000
G1 X30 Y20 Z5 F3000
"""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_string(test_gcode)
    trajectory = vm.get_trajectory()
    
    assert len(trajectory) == 4  # Initial + 3 moves
    assert trajectory[0] == (0.0, 0.0, 0.0)
    assert trajectory[-1] == (30.0, 20.0, 5.0)


def test_vm_microfix_triangular_profile(machine_cfg):
    """Test VM triangular profile time formula."""
    vm = gcode_vm.GCodeVM(machine_cfg, accel_mm_s2=2000.0)
    vm.load_string("G21\nG90\nG1 X1.0 F6000\n")
    r = vm.run()
    assert 0.03 < r['time_estimate_s'] < 0.06


def test_vm_microfix_inches_feed_scaling(machine_cfg):
    """Test VM inches feed scaling."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_string("G20\nG90\nG1 X1.0 F60\n")
    vm.run()
    assert abs(vm.feed - 60*25.4) < 1e-3


def test_vm_microfix_g0_rapid_timing(machine_cfg):
    """Test VM G0 rapid timing is faster than G1."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.rapid_mm_s = 200.0
    vm.load_string("G21\nG90\nG0 X100\n")
    r_rapid = vm.run()
    
    vm.reset()
    vm.load_string("G21\nG90\nG1 X100 F6000\n")
    r_normal = vm.run()
    
    assert r_rapid['time_estimate_s'] < r_normal['time_estimate_s']


def test_vm_microfix_parser_robustness(machine_cfg):
    """Test VM parser accepts X.5 and rejects G10 as move."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_string("G21\nG90\nG1 X.5 F6000\nG10 L2 P1 X0\n")
    r = vm.run()
    assert r['move_count'] == 1
    assert abs(vm.pos[0] - 0.5) < 1e-6


def test_vm_microfix_line_numbers_in_violations(machine_cfg):
    """Test VM includes line numbers in violations."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_string("G21\nG90\nG1 X260 F6000\n")
    r = vm.run()
    assert len(r['violations']) > 0
    assert "line 3" in r['violations'][0]


# ============================================================================
# MLFLOW TESTS
# ============================================================================

def test_mlflow_helpers_graceful_degradation():
    """Test MLflow helpers degrade gracefully when unavailable."""
    if not mlflow_helpers.MLFLOW_AVAILABLE:
        mlflow_helpers.setup_experiment("test_exp")
        with mlflow_helpers.start_run("test_run"):
            mlflow_helpers.log_params_flat({'test': 123})
            mlflow_helpers.log_metrics({'metric': 1.0}, step=0)


@pytest.mark.skipif(not mlflow_helpers.MLFLOW_AVAILABLE, reason="MLflow not installed")
def test_mlflow_helpers_full_functionality(tmp_path):
    """Test MLflow helpers full functionality."""
    params = {
        'agent': {'lr': 0.001},
        'env': {'size': 100}
    }
    mlflow_helpers.log_params_flat(params)
    
    mlflow_helpers.log_resolutions(render_px=908, obs_px=128, reward_px=454)
    
    lut_path = tmp_path / "dummy_lut.pt"
    lut_path.write_bytes(b"dummy data")
    lut_paths = {'color_lut': lut_path}
    mlflow_helpers.log_lut_hashes(lut_paths)
    
    mlflow_helpers.log_reproducibility_info(seed=123, cudnn_deterministic=True)
    
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("test content")
    mlflow_helpers.log_artifact(artifact_path)
    mlflow_helpers.log_artifact_with_hash(artifact_path, log_hash_as_param=True)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_integration_validators_gcode_vm(machine_cfg, tmp_path):
    """Test integration: validators → gcode_gen → vm."""
    strokes_data = {
        'schema': 'stroke.v1',
        'strokes': [
            {
                'id': f'stroke-{i:03d}',
                'bezier': {
                    'p1': [20.0 + i*10, 50.0],
                    'p2': [30.0 + i*10, 60.0],
                    'p3': [40.0 + i*10, 70.0],
                    'p4': [50.0 + i*10, 80.0],
                },
                'z_profile': {'z0': 10.0, 'z1': 12.0},
                'speed_profile': {'v0': 100.0, 'v1': 120.0},
                'color_cmy': {'c': 0.3, 'm': 0.5, 'y': 0.7},
            }
            for i in range(3)
        ]
    }
    
    strokes_obj = validators.StrokesFileV1(**strokes_data)
    assert len(strokes_obj.strokes) == 3
    
    gcode_path = tmp_path / "integration_test.gcode"
    gcode_generator.generate_cmy_gcode(
        strokes_obj, machine_cfg, gcode_path
    )
    assert gcode_path.exists()
    
    vm = gcode_vm.GCodeVM(machine_cfg)
    vm.load_file(gcode_path)
    result = vm.run()
    
    assert len(result['violations']) == 0
    assert result['move_count'] > 3
    assert result['time_estimate_s'] > 0
    
    gcode_content = gcode_path.read_text()
    for i in range(3):
        assert f'STROKE_ID: stroke-{i:03d}' in gcode_content


# ============================================================================
# EDGE CASES
# ============================================================================

def test_edge_case_empty_strokes():
    """Test empty strokes container is valid."""
    empty_strokes = validators.StrokesFileV1(schema='stroke.v1', strokes=[])
    assert len(empty_strokes.strokes) == 0


def test_edge_case_short_stroke_id():
    """Test short stroke ID is rejected."""
    with pytest.raises(ValueError):
        validators.StrokeV1(
            id='abc',
            bezier={'p1': [0,0], 'p2': [1,1], 'p3': [2,2], 'p4': [3,3]},
            z_profile={'z0': 10, 'z1': 10},
            speed_profile={'v0': 100, 'v1': 100},
            color_cmy={'c': 0, 'm': 0, 'y': 0}
        )


def test_edge_case_negative_z():
    """Test negative Z is rejected."""
    with pytest.raises(ValueError):
        validators.ZProfile(z0=-1.0, z1=10.0)


def test_edge_case_z_exceeds_max():
    """Test Z > 30 is rejected."""
    with pytest.raises(ValueError):
        validators.ZProfile(z0=10.0, z1=50.0)


def test_edge_case_speed_below_min():
    """Test speed < 1 is rejected."""
    with pytest.raises(ValueError):
        validators.SpeedProfile(v0=0.5, v1=100.0)


def test_edge_case_cmy_exceeds_max():
    """Test CMY > 1 is rejected."""
    with pytest.raises(ValueError):
        validators.ColorCMY(c=1.5, m=0.5, y=0.5)


def test_edge_case_vm_without_gcode(machine_cfg):
    """Test VM run() fails gracefully without loaded G-code."""
    vm = gcode_vm.GCodeVM(machine_cfg)
    with pytest.raises(RuntimeError):
        vm.run()


def test_hash_and_verify_workflow(tmp_path):
    """Test hash generation and verification workflow."""
    test_file = tmp_path / "data.bin"
    test_file.write_bytes(b"test data")
    
    h1 = hashing.sha256_file(test_file)
    
    test_file.write_bytes(b"modified data")
    h2 = hashing.sha256_file(test_file)
    
    assert h1 != h2
    
    test_file.write_bytes(b"test data")
    h3 = hashing.sha256_file(test_file)
    
    assert h1 == h3


# ============================================================================
# HIGH-VALUE ADDITIONAL TESTS (Critical Production Functionality)
# ============================================================================

def test_apply_tiled_with_padding():
    """Test tiled processing with padding to multiple (critical for GPU memory)."""
    img = torch.rand(1, 3, 250, 250)  # Not multiple of tile size
    
    def identity_fn(x):
        return x
    
    # Test with padding to ensure proper handling
    result = compute.apply_tiled(
        identity_fn, img, tile=128, overlap=16, pad_to_multiple=64
    )
    
    # Should return original size despite padding
    assert result.shape == img.shape
    # Check that the core region (not edges) is preserved
    # Edges may be affected by window blending
    core = img[:, :, 20:-20, 20:-20]
    result_core = result[:, :, 20:-20, 20:-20]
    assert torch.allclose(core, result_core, atol=0.1)


def test_apply_tiled_overlap_blending():
    """Test that tiled processing properly blends overlapping regions."""
    img = torch.ones(1, 3, 256, 256)
    
    def double_fn(x):
        return x * 2.0
    
    result = compute.apply_tiled(double_fn, img, tile=128, overlap=32)
    
    # Should approximately double the values in core region
    # Edges may have blend artifacts
    core = result[:, :, 20:-20, 20:-20]
    assert torch.allclose(core, torch.ones_like(core) * 2.0, atol=0.2)




def test_strokes_heatmap_mm(tmp_path):
    """Test stroke heatmap generation (visualization tool)."""
    # strokes_heatmap_mm expects a list directly, not a dict with 'strokes' key
    strokes_data = [
        {
            'id': 'stroke-001',
            'bezier': {
                'p1': [10.0, 10.0],
                'p2': [20.0, 20.0],
                'p3': [30.0, 30.0],
                'p4': [40.0, 40.0],
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 100.0, 'v1': 100.0},
            'color_cmy': {'c': 0.5, 'm': 0.5, 'y': 0.5},
        },
        {
            'id': 'stroke-002',
            'bezier': {
                'p1': [50.0, 50.0],
                'p2': [60.0, 60.0],
                'p3': [70.0, 70.0],
                'p4': [80.0, 80.0],
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 100.0, 'v1': 100.0},
            'color_cmy': {'c': 0.3, 'm': 0.3, 'y': 0.3},
        }
    ]
    
    yaml_file = tmp_path / "strokes.yaml"
    fs.atomic_yaml_dump(strokes_data, yaml_file)
    
    # Correct signature: H and W as separate parameters
    heatmap = strokes.strokes_heatmap_mm(
        yaml_file,
        work_area_mm=(100.0, 100.0),
        H=100,
        W=100
    )
    
    assert heatmap.shape == (100, 100)
    assert torch.all(heatmap >= 0)
    # Should have some non-zero values where strokes are
    assert torch.sum(heatmap > 0) > 0


def test_filter_strokes_by_bbox():
    """Test filtering strokes by bounding box (spatial culling)."""
    strokes_list = [
        {
            'id': 'stroke-inside',
            'bezier': {
                'p1': [5.0, 5.0],
                'p2': [10.0, 10.0],
                'p3': [20.0, 20.0],
                'p4': [25.0, 25.0]
            }
        },
        {
            'id': 'stroke-outside',
            'bezier': {
                'p1': [100.0, 100.0],
                'p2': [110.0, 110.0],
                'p3': [120.0, 120.0],
                'p4': [130.0, 130.0]
            }
        },
        {
            'id': 'stroke-partial',
            'bezier': {
                'p1': [40.0, 40.0],
                'p2': [45.0, 45.0],
                'p3': [55.0, 55.0],
                'p4': [60.0, 60.0]
            }
        }
    ]
    
    # Filter to region that includes first stroke and partially overlaps third
    filtered = strokes.filter_strokes_by_bbox(
        strokes_list,
        bbox_mm=(0.0, 0.0, 50.0, 50.0)
    )
    
    # Should keep strokes that intersect the bbox
    assert len(filtered) >= 1
    assert any(s['id'] == 'stroke-inside' for s in filtered)


def test_filter_strokes_by_bbox_empty_result():
    """Test filtering strokes when no strokes match bbox."""
    strokes_list = [
        {
            'bezier': {
                'p1': [100.0, 100.0],
                'p2': [110.0, 110.0],
                'p3': [120.0, 120.0],
                'p4': [130.0, 130.0]
            }
        }
    ]
    
    # Filter to region with no strokes
    filtered = strokes.filter_strokes_by_bbox(
        strokes_list,
        bbox_mm=(0.0, 0.0, 10.0, 10.0)
    )
    
    assert len(filtered) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_gpu_mem_info():
    """Test GPU memory information retrieval (critical for OOM debugging)."""
    mem_info = torch_utils.get_gpu_mem_info()
    
    assert 'allocated_mb' in mem_info
    assert 'reserved_mb' in mem_info
    assert 'free_mb' in mem_info
    
    # All memory values should be non-negative
    assert all(v >= 0 for v in mem_info.values())
    
    # Reserved should be >= allocated
    assert mem_info['reserved_mb'] >= mem_info['allocated_mb']


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_cache():
    """Test CUDA cache clearing."""
    # Allocate some memory
    x = torch.rand(1000, 1000, device='cuda')
    del x
    
    # Empty cache should not raise
    torch_utils.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_synchronize():
    """Test CUDA synchronization."""
    # Should not raise
    torch_utils.synchronize()
    
    # After sync, all GPU operations should be complete
    x = torch.rand(100, 100, device='cuda')
    y = x @ x
    torch_utils.synchronize()
    
    # Can safely access result on CPU
    result = y.cpu()
    assert result.shape == (100, 100)




def test_compute_all_metrics():
    """Test computing all metrics at once (integration utility)."""
    target = torch.rand(3, 64, 64)
    canvas = target + torch.randn(3, 64, 64) * 0.1
    canvas = torch.clamp(canvas, 0, 1)
    
    all_metrics = metrics.compute_all_metrics(target, canvas)
    
    # Should return a dict with all standard metrics
    assert 'psnr' in all_metrics
    assert 'ssim' in all_metrics
    
    # Values should be reasonable
    assert all_metrics['psnr'] > 0
    assert 0 <= all_metrics['ssim'] <= 1.0


def test_clamp_finite_replaces_nan():
    """Test clamp_finite replaces NaN with clamped values."""
    x = torch.tensor([1.0, float('nan'), 0.5, float('inf'), -float('inf')])
    
    # Correct signature: min_val and max_val
    clamped = compute.clamp_finite(x, min_val=0.0, max_val=1.0)
    
    # No NaN or Inf should remain
    assert not torch.isnan(clamped).any()
    assert not torch.isinf(clamped).any()
    
    # All values should be in [0, 1]
    assert torch.all(clamped >= 0.0)
    assert torch.all(clamped <= 1.0)
    
    # Regular values should be unchanged
    assert clamped[0] == 1.0
    assert clamped[2] == 0.5


def test_autocast_context_bf16():
    """Test autocast context manager with bfloat16."""
    with compute.autocast_context(precision="bf16"):
        x = torch.rand(10, 10)
        y = x @ x
        # Inside autocast, operations may use lower precision
        # But result should still be valid
        assert y.shape == (10, 10)


def test_autocast_context_fp16():
    """Test autocast context manager with float16."""
    with compute.autocast_context(precision="fp16"):
        x = torch.rand(10, 10)
        y = x @ x
        assert y.shape == (10, 10)


def test_autocast_context_fp32():
    """Test autocast context manager with float32 (no casting)."""
    with compute.autocast_context(precision="fp32"):
        x = torch.rand(10, 10)
        y = x @ x
        assert y.dtype == torch.float32


def test_print_model_summary():
    """Test model summary printing (should not crash)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Should not raise
    torch_utils.print_model_summary(model, name="TestModel")


def test_no_autocast_context():
    """Test no_autocast context manager."""
    with torch_utils.no_autocast():
        x = torch.rand(10, 10)
        y = x @ x
        # Should always use float32
        assert y.dtype == torch.float32


def test_tile_slices_large_image():
    """Test tile slice generation for large images."""
    slices = compute.tile_slices(H=1024, W=1024, tile=256, overlap=32)
    
    # Should generate reasonable number of tiles
    assert len(slices) > 0
    
    # Each slice should be within bounds
    for (h_slice, w_slice) in slices:
        assert h_slice.stop <= 1024
        assert w_slice.stop <= 1024
        assert h_slice.start >= 0
        assert w_slice.start >= 0


def test_choose_tile_size_from_mem():
    """Test automatic tile size selection based on available memory."""
    # Test with various image sizes - correct signature: bytes_per_px and mem_margin
    tile_size = compute.choose_tile_size_from_mem(
        H=2048, W=2048, 
        bytes_per_px=16,
        mem_margin=0.5
    )
    
    # Should return a reasonable tile size (or 0 for full frame)
    assert tile_size >= 0
    if tile_size > 0:
        assert tile_size <= 2048


def test_validate_stroke_cap(tmp_path):
    """Test stroke cap validation."""
    # count_strokes expects a list directly, not a dict
    strokes_data = [
        {
            'id': f'stroke-{i:03d}',
            'bezier': {
                'p1': [i, i],
                'p2': [i+1, i+1],
                'p3': [i+2, i+2],
                'p4': [i+3, i+3]
            }
        } for i in range(100)
    ]
    yaml_file = tmp_path / "strokes.yaml"
    fs.atomic_yaml_dump(strokes_data, yaml_file)
    
    # Should pass with cap of 150 (returns True)
    result = strokes.validate_stroke_cap(yaml_file, stroke_cap=150)
    assert result is True
    
    # Should return False with cap of 50
    result = strokes.validate_stroke_cap(yaml_file, stroke_cap=50)
    assert result is False


def test_image_mm_to_machine_mm_no_flip():
    """Test coordinate transform without Y-flip."""
    pts_img = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    work_area = (100.0, 100.0)
    
    pts_mach = compute.image_mm_to_machine_mm(pts_img, work_area, flip_y=False)
    
    # Without flip, should be identity transform
    assert torch.allclose(pts_img, pts_mach)


def test_image_mm_to_machine_mm_with_flip():
    """Test coordinate transform with Y-flip."""
    pts_img = torch.tensor([[10.0, 20.0]])
    work_area = (100.0, 100.0)
    
    pts_mach = compute.image_mm_to_machine_mm(pts_img, work_area, flip_y=True)
    
    # X should be unchanged, Y should be flipped
    assert pts_mach[0, 0] == 10.0
    assert pts_mach[0, 1] == 80.0  # 100 - 20
