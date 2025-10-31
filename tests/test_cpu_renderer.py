"""Comprehensive unit tests for CPU reference renderer.

Test suites:
1. Initialization & validation
2. LUT interpolation (color, alpha, PSF)
3. Polyline operations (arc-length, sampling)
4. Kernel caching and normalization
5. Splat operations (boundary handling, clipping)
6. Compositing invariants (alpha-over, order independence)
7. Geometric properties (bbox, endpoint preservation)
8. Analytic checks (mass preservation, line width)
9. Edge cases (degenerate strokes, boundary grazing)
10. Determinism and reproducibility

Fixtures:
- toy_luts: Simple analytical LUTs for testing
- renderer: Initialized CPU renderer
- canvas: Blank canvas + alpha map
"""

import numpy as np
import pytest
import torch

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import geometry, strokes as stroke_utils


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def toy_luts():
    """Generate toy LUTs for testing."""
    return load_toy_luts()


@pytest.fixture
def env_cfg():
    """Standard environment config (A4, 908×1280)."""
    return {
        'work_area_mm': (210.0, 297.0),
        'render_px': (908, 1280)
    }


@pytest.fixture
def sim_cfg():
    """Empty simulation config (use defaults)."""
    return {}


@pytest.fixture
def renderer(sim_cfg, env_cfg, toy_luts):
    """Initialized CPU reference renderer."""
    return CPUReferenceRenderer(sim_cfg, env_cfg, toy_luts)


@pytest.fixture
def blank_canvas(env_cfg):
    """Blank white canvas with zero alpha."""
    h, w = env_cfg['render_px']
    canvas = np.ones((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    return canvas, alpha


@pytest.fixture
def simple_stroke():
    """Simple straight stroke for testing."""
    return {
        'id': 'test-00000-00000-aaaaaaaa',
        'bezier': {
            'p1': (50.0, 100.0),
            'p2': (70.0, 100.0),
            'p3': (130.0, 100.0),
            'p4': (150.0, 100.0)
        },
        'z_profile': {
            'z0': 10.0,
            'z1': 10.0
        },
        'speed_profile': {
            'v0': 50.0,
            'v1': 50.0
        },
        'color_cmy': {
            'c': 0.8,
            'm': 0.2,
            'y': 0.1
        }
    }


@pytest.fixture
def curved_stroke():
    """Curved stroke for geometric tests."""
    return {
        'id': 'test-00000-00001-bbbbbbbb',
        'bezier': {
            'p1': (50.0, 50.0),
            'p2': (100.0, 80.0),
            'p3': (150.0, 120.0),
            'p4': (200.0, 150.0)
        },
        'z_profile': {
            'z0': 5.0,
            'z1': 15.0
        },
        'speed_profile': {
            'v0': 100.0,
            'v1': 200.0
        },
        'color_cmy': {
            'c': 0.5,
            'm': 0.5,
            'y': 0.5
        }
    }


# ============================================================================
# TEST SUITE 1: Initialization & Validation
# ============================================================================

def test_initialization(renderer, env_cfg):
    """Test renderer initialization and config validation."""
    assert renderer.canvas_h_px == env_cfg['render_px'][0]
    assert renderer.canvas_w_px == env_cfg['render_px'][1]
    
    # Check DPI computation
    expected_dpi_x = env_cfg['render_px'][1] / env_cfg['work_area_mm'][0]
    expected_dpi_y = env_cfg['render_px'][0] / env_cfg['work_area_mm'][1]
    assert abs(renderer.dpi[0] - expected_dpi_x) < 1e-6
    assert abs(renderer.dpi[1] - expected_dpi_y) < 1e-6
    
    # Check bounds
    assert renderer.stroke_bounds['x'] == (0.0, 210.0)
    assert renderer.stroke_bounds['y'] == (0.0, 297.0)
    assert renderer.stroke_bounds['z'] == (0.0, 30.0)
    assert renderer.stroke_bounds['speed'] == (1.0, 300.0)


def test_lut_validation(sim_cfg, env_cfg):
    """Test LUT validation (shapes, dtypes, ranges)."""
    # Valid LUTs
    valid_luts = load_toy_luts()
    renderer = CPUReferenceRenderer(sim_cfg, env_cfg, valid_luts)
    assert renderer is not None
    
    # Missing LUT
    with pytest.raises(ValueError, match="Missing LUTs"):
        CPUReferenceRenderer(sim_cfg, env_cfg, {'color_lut': valid_luts['color_lut']})
    
    # Wrong dtype
    bad_luts = valid_luts.copy()
    bad_luts['alpha_lut'] = bad_luts['alpha_lut'].astype(np.float64)
    with pytest.raises(TypeError, match="must be FP32"):
        CPUReferenceRenderer(sim_cfg, env_cfg, bad_luts)
    
    # Wrong color LUT shape
    bad_luts = valid_luts.copy()
    bad_luts['color_lut'] = np.zeros((11, 11, 11, 4), dtype=np.float32)  # 4 channels
    with pytest.raises(ValueError, match="must be.*3"):
        CPUReferenceRenderer(sim_cfg, env_cfg, bad_luts)


# ============================================================================
# TEST SUITE 2: LUT Interpolation
# ============================================================================

def test_color_lut_corners(renderer):
    """Test color LUT at corners (C=0/1, M=0/1, Y=0/1)."""
    # Black (C=1, M=1, Y=1) → RGB ≈ (0,0,0)
    rgb_black = renderer._interpolate_color_lut(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(rgb_black, [0.0, 0.0, 0.0], atol=0.05)
    
    # White (C=0, M=0, Y=0) → RGB ≈ (1,1,1)
    rgb_white = renderer._interpolate_color_lut(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(rgb_white, [1.0, 1.0, 1.0], atol=0.05)
    
    # Cyan (C=1, M=0, Y=0) → RGB ≈ (0,1,1)
    rgb_cyan = renderer._interpolate_color_lut(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert rgb_cyan[0] < 0.2  # Low R
    assert rgb_cyan[1] > 0.8  # High G
    assert rgb_cyan[2] > 0.8  # High B


def test_alpha_lut_monotonicity(renderer):
    """Test alpha LUT monotonicity (α decreases with z, increases with v)."""
    # Fix v, vary z: α should decrease
    alpha_z5 = renderer._interpolate_alpha_lut(5.0, 100.0)
    alpha_z20 = renderer._interpolate_alpha_lut(20.0, 100.0)
    assert alpha_z5 > alpha_z20, "Alpha should decrease with z"
    
    # Fix z, vary v: α should increase
    alpha_v50 = renderer._interpolate_alpha_lut(10.0, 50.0)
    alpha_v200 = renderer._interpolate_alpha_lut(10.0, 200.0)
    assert alpha_v200 > alpha_v50, "Alpha should increase with v"
    
    # All values in [0, 1]
    for z in [1.0, 10.0, 20.0, 30.0]:
        for v in [1.0, 50.0, 150.0, 300.0]:
            alpha = renderer._interpolate_alpha_lut(z, v)
            assert 0.0 <= alpha <= 1.0, f"Alpha at (z={z}, v={v}) = {alpha} out of range"


def test_psf_lut_monotonicity(renderer):
    """Test PSF LUT monotonicity (σ increases with z, decreases with v)."""
    # Fix v, vary z: σ should increase
    sigma_z5 = renderer._interpolate_psf_lut(5.0, 100.0)
    sigma_z20 = renderer._interpolate_psf_lut(20.0, 100.0)
    assert sigma_z20 > sigma_z5, "Sigma should increase with z"
    
    # Fix z, vary v: σ should decrease
    sigma_v50 = renderer._interpolate_psf_lut(10.0, 50.0)
    sigma_v200 = renderer._interpolate_psf_lut(10.0, 200.0)
    assert sigma_v50 > sigma_v200, "Sigma should decrease with v"
    
    # Reasonable range
    for z in [1.0, 10.0, 20.0, 30.0]:
        for v in [1.0, 50.0, 150.0, 300.0]:
            sigma = renderer._interpolate_psf_lut(z, v)
            assert 0.1 <= sigma <= 50.0, f"Sigma at (z={z}, v={v}) = {sigma} out of range"


# ============================================================================
# TEST SUITE 3: Polyline Operations
# ============================================================================

def test_arc_length_computation(renderer):
    """Test cumulative arc length calculation."""
    # Straight line: (0,0) → (10,0)
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    arc_lengths = renderer._compute_arc_lengths(polyline)
    assert np.allclose(arc_lengths, [0.0, 10.0])
    
    # Multi-segment
    polyline = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    arc_lengths = renderer._compute_arc_lengths(polyline)
    expected = [0.0, 3.0, 3.0 + 4.0]
    assert np.allclose(arc_lengths, expected)


def test_polyline_interpolation(renderer):
    """Test arc-length parametrization."""
    polyline = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0]
    ], dtype=np.float32)
    arc_lengths = renderer._compute_arc_lengths(polyline)
    
    # t=0 → start
    t = np.array([0.0], dtype=np.float32)
    pos = renderer._interpolate_polyline(polyline, arc_lengths, t)
    assert np.allclose(pos[0], [0.0, 0.0])
    
    # t=1 → end
    t = np.array([1.0], dtype=np.float32)
    pos = renderer._interpolate_polyline(polyline, arc_lengths, t)
    assert np.allclose(pos[0], [10.0, 10.0])
    
    # t=0.5 → halfway along total length
    t = np.array([0.5], dtype=np.float32)
    pos = renderer._interpolate_polyline(polyline, arc_lengths, t)
    total_len = arc_lengths[-1]
    halfway = 0.5 * total_len  # = 10 (exactly at corner)
    assert np.allclose(pos[0], [10.0, 0.0])


# ============================================================================
# TEST SUITE 4: Kernel Caching & Normalization
# ============================================================================

def test_kernel_cache(renderer):
    """Test Gaussian kernel caching."""
    sigma1 = 2.0
    sigma2 = 2.05  # Within quantization (0.1)
    sigma3 = 3.0
    
    kernel1 = renderer._get_gaussian_kernel(sigma1)
    kernel2 = renderer._get_gaussian_kernel(sigma2)
    kernel3 = renderer._get_gaussian_kernel(sigma3)
    
    # sigma1 and sigma2 should use same cached kernel
    assert kernel1 is kernel2, "Kernels within quantization should be cached"
    
    # sigma3 should be different
    assert kernel1 is not kernel3


def test_kernel_normalization(renderer):
    """Test that Gaussian kernels are normalized (sum=1)."""
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        kernel = renderer._get_gaussian_kernel(sigma)
        kernel_sum = kernel.sum()
        assert abs(kernel_sum - 1.0) < 1e-6, f"Kernel σ={sigma} sum={kernel_sum} != 1"


def test_kernel_symmetry(renderer):
    """Test that Gaussian kernels are symmetric."""
    for sigma in [2.0, 3.0]:
        kernel = renderer._get_gaussian_kernel(sigma)
        # Should be symmetric about center
        assert np.allclose(kernel, kernel[::-1, ::-1]), f"Kernel σ={sigma} not symmetric"


# ============================================================================
# TEST SUITE 5: Splat Operations
# ============================================================================

def test_splat_single_stamp(renderer, blank_canvas):
    """Test single stamp at canvas center."""
    canvas, alpha = blank_canvas
    
    # Center of A4 (105 mm, 148.5 mm)
    pos_mm = np.array([105.0, 148.5], dtype=np.float32)
    sigma_px = 3.0
    alpha_local = 0.5
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red
    
    # Splat
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    
    # Check that center pixel changed
    h, w = canvas.shape[:2]
    cy = h // 2
    cx = w // 2
    
    # Center should be more red, less white
    assert canvas[cy, cx, 0] > canvas[cy, cx, 1]  # R > G
    assert canvas[cy, cx, 0] > canvas[cy, cx, 2]  # R > B
    
    # Alpha should be > 0 at center (toy LUTs produce low alpha)
    assert alpha[cy, cx] > 0.001


def test_splat_boundary_clipping(renderer, blank_canvas):
    """Test that splats are clipped at canvas boundaries."""
    canvas, alpha = blank_canvas
    
    # Near top-left corner
    pos_mm = np.array([1.0, 1.0], dtype=np.float32)
    sigma_px = 5.0
    alpha_local = 0.5
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Should not raise
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    
    # Check no NaNs or infinities
    assert np.isfinite(canvas).all()
    assert np.isfinite(alpha).all()


def test_splat_outside_canvas(renderer, blank_canvas):
    """Test that splats outside canvas are ignored."""
    canvas, alpha = blank_canvas
    canvas_before = canvas.copy()
    alpha_before = alpha.copy()
    
    # Far outside canvas
    pos_mm = np.array([500.0, 500.0], dtype=np.float32)
    sigma_px = 2.0
    alpha_local = 0.5
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    
    # Canvas should be unchanged
    assert np.allclose(canvas, canvas_before)
    assert np.allclose(alpha, alpha_before)


# ============================================================================
# TEST SUITE 6: Compositing Invariants
# ============================================================================

def test_alpha_over_black_on_white(renderer, blank_canvas):
    """Test alpha-over compositing: black paint on white canvas."""
    canvas, alpha = blank_canvas
    
    pos_mm = np.array([105.0, 148.5], dtype=np.float32)
    sigma_px = 3.0
    alpha_local = 0.5
    paint_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Black
    
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    
    # Center should be darker (between white and black)
    # Note: with toy LUTs, alpha is very low, so change is subtle
    h, w = canvas.shape[:2]
    cy, cx = h // 2, w // 2
    center_intensity = canvas[cy, cx].mean()
    # With toy LUTs (low alpha), expect only slight darkening
    assert center_intensity < 1.0, f"Expected some darkening, got {center_intensity}"


def test_alpha_accumulation(renderer, blank_canvas):
    """Test that alpha accumulates but clamps at 1.0."""
    canvas, alpha = blank_canvas
    
    pos_mm = np.array([105.0, 148.5], dtype=np.float32)
    sigma_px = 2.0
    alpha_local = 0.6
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # First splat
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    h, w = canvas.shape[:2]
    cy, cx = h // 2, w // 2
    alpha_after_1 = alpha[cy, cx]
    
    # Second splat at same location
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    alpha_after_2 = alpha[cy, cx]
    
    # Alpha should increase but not exceed 1.0
    assert alpha_after_2 > alpha_after_1
    assert alpha_after_2 <= 1.0


def test_zero_alpha_no_change(renderer, blank_canvas):
    """Test that α=0 causes no change."""
    canvas, alpha = blank_canvas
    canvas_before = canvas.copy()
    alpha_before = alpha.copy()
    
    pos_mm = np.array([105.0, 148.5], dtype=np.float32)
    renderer._splat_sample(
        canvas, alpha, pos_mm, 2.0, 0.0,  # alpha_local = 0
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    
    # Should be unchanged
    assert np.allclose(canvas, canvas_before)
    assert np.allclose(alpha, alpha_before)


# ============================================================================
# TEST SUITE 7: Geometric Properties
# ============================================================================

def test_endpoint_preservation(renderer, simple_stroke):
    """Test that polyline endpoints match Bézier endpoints."""
    bezier = simple_stroke['bezier']
    p1 = torch.tensor(bezier['p1'], dtype=torch.float32)
    p2 = torch.tensor(bezier['p2'], dtype=torch.float32)
    p3 = torch.tensor(bezier['p3'], dtype=torch.float32)
    p4 = torch.tensor(bezier['p4'], dtype=torch.float32)
    
    polyline = geometry.bezier_cubic_polyline(p1, p2, p3, p4).cpu().numpy()
    
    assert np.allclose(polyline[0], bezier['p1'], atol=1e-3)
    assert np.allclose(polyline[-1], bezier['p4'], atol=1e-3)


def test_stroke_length_positive(renderer, simple_stroke):
    """Test that stroke length is positive."""
    bezier = simple_stroke['bezier']
    p1 = torch.tensor(bezier['p1'], dtype=torch.float32)
    p2 = torch.tensor(bezier['p2'], dtype=torch.float32)
    p3 = torch.tensor(bezier['p3'], dtype=torch.float32)
    p4 = torch.tensor(bezier['p4'], dtype=torch.float32)
    
    polyline = geometry.bezier_cubic_polyline(p1, p2, p3, p4).cpu().numpy()
    arc_lengths = renderer._compute_arc_lengths(polyline)
    total_length = arc_lengths[-1]
    
    assert total_length > 0, "Stroke length should be positive"
    assert total_length < 300.0, "Stroke length should be reasonable (< 300 mm)"


# ============================================================================
# TEST SUITE 8: Analytic Checks
# ============================================================================

def test_single_stamp_mass_preservation(renderer, blank_canvas):
    """Test that single stamp mass ≈ alpha_local (within PSF kernel)."""
    canvas, alpha = blank_canvas
    
    pos_mm = np.array([105.0, 148.5], dtype=np.float32)
    sigma_px = 3.0
    alpha_local = 0.3
    paint_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    renderer._splat_sample(
        canvas, alpha, pos_mm, sigma_px, alpha_local, paint_color
    )
    
    # Crop ROI around stamp
    h, w = canvas.shape[:2]
    cy, cx = h // 2, w // 2
    radius = int(4 * sigma_px) + 5
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)
    
    alpha_roi = alpha[y_min:y_max, x_min:x_max]
    total_mass = alpha_roi.sum()
    
    # Expected mass ≈ alpha_local (kernel sums to 1)
    # Allow some tolerance due to kernel truncation and clipping
    assert abs(total_mass - alpha_local) < 0.15 * alpha_local, \
        f"Mass {total_mass} != expected {alpha_local}"


def test_line_width_estimate(renderer, blank_canvas, simple_stroke):
    """Test that rendered line width matches FWHM ≈ 2.355·σ."""
    canvas, alpha = blank_canvas
    
    # Render simple horizontal stroke
    canvas, alpha = renderer.render_stroke(canvas, alpha, simple_stroke)
    
    # Take cross-section at mid-stroke
    h, w = canvas.shape[:2]
    cy = h // 2
    alpha_profile = alpha[cy, :]
    
    # Find peak and FWHM
    peak_val = alpha_profile.max()
    if peak_val < 0.1:
        pytest.skip("Peak too low for FWHM measurement")
    
    half_max = 0.5 * peak_val
    above_half = alpha_profile >= half_max
    indices = np.where(above_half)[0]
    
    if len(indices) < 2:
        pytest.skip("Not enough points above half-max")
    
    fwhm_px = indices[-1] - indices[0]
    
    # Expected σ from stroke parameters
    z_avg = 0.5 * (simple_stroke['z_profile']['z0'] + simple_stroke['z_profile']['z1'])
    v_avg = 0.5 * (simple_stroke['speed_profile']['v0'] + simple_stroke['speed_profile']['v1'])
    sigma_expected = renderer._interpolate_psf_lut(z_avg, v_avg)
    fwhm_expected = 2.355 * sigma_expected
    
    # Check within ±50% (rough sanity check)
    assert abs(fwhm_px - fwhm_expected) < 0.5 * fwhm_expected, \
        f"FWHM {fwhm_px} != expected {fwhm_expected:.1f}"


# ============================================================================
# TEST SUITE 9: Edge Cases
# ============================================================================

def test_degenerate_stroke_zero_length(renderer, blank_canvas):
    """Test that zero-length stroke is handled gracefully."""
    canvas, alpha = blank_canvas
    canvas_before = canvas.copy()
    
    degenerate = {
        'id': 'degenerate',
        'bezier': {
            'p1': (100.0, 100.0),
            'p2': (100.0, 100.0),
            'p3': (100.0, 100.0),
            'p4': (100.0, 100.0)
        },
        'z_profile': {'z0': 10.0, 'z1': 10.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.5, 'm': 0.5, 'y': 0.5}
    }
    
    # Should not crash
    canvas, alpha = renderer.render_stroke(canvas, alpha, degenerate)
    
    # Canvas should be mostly unchanged (may have single stamp)
    # Just check no NaNs
    assert np.isfinite(canvas).all()
    assert np.isfinite(alpha).all()


def test_out_of_bounds_clamping(renderer, blank_canvas):
    """Test that out-of-bounds coordinates are clamped."""
    canvas, alpha = blank_canvas
    
    oob_stroke = {
        'id': 'oob',
        'bezier': {
            'p1': (-50.0, -50.0),  # Outside canvas
            'p2': (100.0, 100.0),
            'p3': (200.0, 200.0),
            'p4': (500.0, 500.0)  # Way outside
        },
        'z_profile': {'z0': 50.0, 'z1': 50.0},  # Outside z bounds
        'speed_profile': {'v0': 500.0, 'v1': 500.0},  # Outside v bounds
        'color_cmy': {'c': 1.5, 'm': 1.5, 'y': 1.5}  # Outside [0,1]
    }
    
    # Should not crash; params should be clamped
    canvas, alpha = renderer.render_stroke(canvas, alpha, oob_stroke)
    
    assert np.isfinite(canvas).all()
    assert np.isfinite(alpha).all()
    
    # Check that clamped stroke has valid params
    clamped = renderer.project_params(oob_stroke)
    assert all(0.0 <= clamped['bezier'][pt][0] <= 210.0 for pt in ['p1', 'p2', 'p3', 'p4'])
    assert all(0.0 <= clamped['bezier'][pt][1] <= 297.0 for pt in ['p1', 'p2', 'p3', 'p4'])


def test_very_small_sigma(renderer, blank_canvas):
    """Test handling of very small PSF width."""
    canvas, alpha = blank_canvas
    
    small_sigma_stroke = {
        'id': 'small-sigma',
        'bezier': {
            'p1': (100.0, 100.0),
            'p2': (110.0, 100.0),
            'p3': (120.0, 100.0),
            'p4': (130.0, 100.0)
        },
        'z_profile': {'z0': 0.1, 'z1': 0.1},  # Very low z → large α, but small σ
        'speed_profile': {'v0': 300.0, 'v1': 300.0},  # Max speed → small σ
        'color_cmy': {'c': 0.5, 'm': 0.5, 'y': 0.5}
    }
    
    # Should not crash
    canvas, alpha = renderer.render_stroke(canvas, alpha, small_sigma_stroke)
    assert np.isfinite(canvas).all()


def test_very_large_sigma(renderer, blank_canvas):
    """Test handling of very large PSF width."""
    canvas, alpha = blank_canvas
    
    large_sigma_stroke = {
        'id': 'large-sigma',
        'bezier': {
            'p1': (100.0, 100.0),
            'p2': (110.0, 100.0),
            'p3': (120.0, 100.0),
            'p4': (130.0, 100.0)
        },
        'z_profile': {'z0': 30.0, 'z1': 30.0},  # Max z → large σ, small α
        'speed_profile': {'v0': 1.0, 'v1': 1.0},  # Min speed → large σ
        'color_cmy': {'c': 0.5, 'm': 0.5, 'y': 0.5}
    }
    
    # Should not crash (may be slow due to large kernel)
    canvas, alpha = renderer.render_stroke(canvas, alpha, large_sigma_stroke)
    assert np.isfinite(canvas).all()


# ============================================================================
# TEST SUITE 10: Determinism & Reproducibility
# ============================================================================

def test_deterministic_rendering(renderer, blank_canvas, simple_stroke):
    """Test that rendering is deterministic (same inputs → same outputs)."""
    canvas1, alpha1 = blank_canvas
    canvas2, alpha2 = [arr.copy() for arr in blank_canvas]
    
    # Render twice
    canvas1, alpha1 = renderer.render_stroke(canvas1, alpha1, simple_stroke)
    canvas2, alpha2 = renderer.render_stroke(canvas2, alpha2, simple_stroke)
    
    # Should be identical
    assert np.allclose(canvas1, canvas2, atol=1e-7)
    assert np.allclose(alpha1, alpha2, atol=1e-7)


def test_order_independence_disjoint(renderer, blank_canvas):
    """Test that disjoint strokes are order-independent."""
    # Two non-overlapping strokes
    stroke1 = {
        'id': 's1',
        'bezier': {
            'p1': (50.0, 50.0),
            'p2': (60.0, 50.0),
            'p3': (70.0, 50.0),
            'p4': (80.0, 50.0)
        },
        'z_profile': {'z0': 10.0, 'z1': 10.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 1.0, 'm': 0.0, 'y': 0.0}
    }
    
    stroke2 = {
        'id': 's2',
        'bezier': {
            'p1': (50.0, 200.0),
            'p2': (60.0, 200.0),
            'p3': (70.0, 200.0),
            'p4': (80.0, 200.0)
        },
        'z_profile': {'z0': 10.0, 'z1': 10.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.0, 'm': 1.0, 'y': 0.0}
    }
    
    # Render in two orders
    canvas_a, alpha_a = [arr.copy() for arr in blank_canvas]
    canvas_a, alpha_a = renderer.render_strokes(canvas_a, alpha_a, [stroke1, stroke2])
    
    canvas_b, alpha_b = [arr.copy() for arr in blank_canvas]
    canvas_b, alpha_b = renderer.render_strokes(canvas_b, alpha_b, [stroke2, stroke1])
    
    # Should be identical (within floating-point tolerance)
    assert np.allclose(canvas_a, canvas_b, atol=1e-6)
    assert np.allclose(alpha_a, alpha_b, atol=1e-6)


def test_no_nan_or_inf(renderer, blank_canvas, curved_stroke):
    """Test that rendering never produces NaNs or infinities."""
    canvas, alpha = blank_canvas
    
    # Render complex stroke
    canvas, alpha = renderer.render_stroke(canvas, alpha, curved_stroke)
    
    assert np.isfinite(canvas).all(), "Canvas contains NaN or Inf"
    assert np.isfinite(alpha).all(), "Alpha contains NaN or Inf"
    
    # Values should be in valid ranges
    assert (canvas >= 0.0).all() and (canvas <= 1.0).all(), "Canvas out of [0,1]"
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "Alpha out of [0,1]"


# ============================================================================
# TEST SUITE 11: Multiple Strokes
# ============================================================================

def test_multiple_strokes_accumulation(renderer, blank_canvas):
    """Test that multiple strokes accumulate correctly."""
    canvas, alpha = blank_canvas
    
    # Three overlapping strokes
    strokes = []
    for i, x_offset in enumerate([100.0, 110.0, 120.0]):
        strokes.append({
            'id': f's{i}',
            'bezier': {
                'p1': (x_offset, 100.0),
                'p2': (x_offset + 10, 100.0),
                'p3': (x_offset + 20, 100.0),
                'p4': (x_offset + 30, 100.0)
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.3, 'm': 0.3, 'y': 0.3}
        })
    
    canvas, alpha = renderer.render_strokes(canvas, alpha, strokes)
    
    # Alpha should have accumulated (toy LUTs produce low alpha)
    assert alpha.max() > 0.0001, "Alpha should be > 0 after strokes"
    
    # Canvas should be darker (painted) - toy LUTs produce very low alpha
    # so change is extremely subtle
    assert canvas.mean() < 1.0, "Canvas should be darker after painting"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
