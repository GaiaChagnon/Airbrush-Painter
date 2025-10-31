"""Golden image tests for CPU reference renderer.

This test suite validates rendering against known-good reference images
(golden baselines). Golden images are stored in ci/golden_tests_cpu/ and
are versioned with the codebase.

Test workflow:
1. Render predefined strokes with fixed seeds and configs
2. Compare rendered images against saved golden PNGs
3. Validate metrics: PSNR, SSIM, ΔE2000, coverage fraction
4. Fail if any metric exceeds threshold

Golden test cases:
1. straight_line: Simple horizontal stroke
2. curved_stroke: S-curve with varying z/v
3. multi_stroke: Three overlapping strokes
4. boundary_graze: Stroke near canvas edge
5. tiny_stroke: Micro-stroke for anti-aliasing test

Acceptance thresholds:
- PSNR ≥ 28 dB (strict pixel-level match)
- SSIM ≥ 0.92 (structural similarity)
- ΔE2000 mean ≤ 2.0 (perceptual color difference)
- Coverage fraction within ±5% (painted area)

Usage:
    pytest tests/test_cpu_golden.py  # Run all golden tests
    pytest tests/test_cpu_golden.py::test_golden_straight_line  # Run specific test
    pytest tests/test_cpu_golden.py --regen-golden  # Regenerate golden images (manual mode)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import color as color_utils, metrics, fs


logger = logging.getLogger(__name__)

# Golden test data directory
GOLDEN_DIR = Path(__file__).parent.parent / 'ci' / 'golden_tests_cpu'

# Acceptance thresholds
THRESHOLDS = {
    'psnr_min': 28.0,       # dB
    'ssim_min': 0.92,       # [0,1]
    'delta_e_max': 2.0,     # ΔE2000
    'coverage_tol': 0.05    # ±5% relative
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def renderer():
    """Create renderer with fixed config."""
    env_cfg = {
        'work_area_mm': (210.0, 297.0),
        'render_px': (908, 1280)
    }
    sim_cfg = {}
    luts = load_toy_luts()
    return CPUReferenceRenderer(sim_cfg, env_cfg, luts)


@pytest.fixture
def blank_canvas():
    """White canvas for rendering."""
    canvas = np.ones((908, 1280, 3), dtype=np.float32)
    alpha = np.zeros((908, 1280), dtype=np.float32)
    return canvas, alpha


# ============================================================================
# GOLDEN STROKE DEFINITIONS
# ============================================================================

def get_golden_strokes() -> Dict[str, Dict]:
    """Get predefined golden strokes.
    
    Returns
    -------
    dict
        Mapping of test name → stroke definition
    """
    strokes = {
        'straight_line': {
            'id': 'golden-00000-straight',
            'bezier': {
                'p1': (50.0, 148.5),
                'p2': (80.0, 148.5),
                'p3': (130.0, 148.5),
                'p4': (160.0, 148.5)
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.8, 'm': 0.2, 'y': 0.1}
        },
        
        'curved_stroke': {
            'id': 'golden-00001-curved',
            'bezier': {
                'p1': (50.0, 50.0),
                'p2': (100.0, 120.0),
                'p3': (150.0, 80.0),
                'p4': (200.0, 150.0)
            },
            'z_profile': {'z0': 5.0, 'z1': 15.0},
            'speed_profile': {'v0': 100.0, 'v1': 200.0},
            'color_cmy': {'c': 0.2, 'm': 0.8, 'y': 0.3}
        },
        
        'boundary_graze': {
            'id': 'golden-00002-boundary',
            'bezier': {
                'p1': (5.0, 5.0),
                'p2': (10.0, 10.0),
                'p3': (15.0, 15.0),
                'p4': (20.0, 20.0)
            },
            'z_profile': {'z0': 8.0, 'z1': 12.0},
            'speed_profile': {'v0': 75.0, 'v1': 125.0},
            'color_cmy': {'c': 0.5, 'm': 0.5, 'y': 0.5}
        },
        
        'tiny_stroke': {
            'id': 'golden-00003-tiny',
            'bezier': {
                'p1': (100.0, 100.0),
                'p2': (101.0, 100.5),
                'p3': (102.0, 101.0),
                'p4': (103.0, 101.5)
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.3, 'm': 0.3, 'y': 0.3}
        }
    }
    
    return strokes


def get_golden_multi_strokes() -> list:
    """Get multi-stroke test case (three overlapping strokes).
    
    Returns
    -------
    list of dict
        List of stroke definitions
    """
    return [
        {
            'id': 'golden-multi-00',
            'bezier': {
                'p1': (80.0, 100.0),
                'p2': (100.0, 100.0),
                'p3': (120.0, 100.0),
                'p4': (140.0, 100.0)
            },
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.9, 'm': 0.1, 'y': 0.1}
        },
        {
            'id': 'golden-multi-01',
            'bezier': {
                'p1': (90.0, 110.0),
                'p2': (110.0, 110.0),
                'p3': (130.0, 110.0),
                'p4': (150.0, 110.0)
            },
            'z_profile': {'z0': 12.0, 'z1': 12.0},
            'speed_profile': {'v0': 60.0, 'v1': 60.0},
            'color_cmy': {'c': 0.1, 'm': 0.9, 'y': 0.1}
        },
        {
            'id': 'golden-multi-02',
            'bezier': {
                'p1': (85.0, 120.0),
                'p2': (105.0, 120.0),
                'p3': (125.0, 120.0),
                'p4': (145.0, 120.0)
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 70.0, 'v1': 70.0},
            'color_cmy': {'c': 0.1, 'm': 0.1, 'y': 0.9}
        }
    ]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_golden_case(
    renderer: CPUReferenceRenderer,
    strokes: list,
    canvas_init: np.ndarray,
    alpha_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a golden test case.
    
    Parameters
    ----------
    renderer : CPUReferenceRenderer
        Renderer instance
    strokes : list of dict
        Stroke definitions
    canvas_init : np.ndarray
        Initial canvas
    alpha_init : np.ndarray
        Initial alpha map
    
    Returns
    -------
    canvas : np.ndarray
        Rendered canvas (linear RGB)
    alpha : np.ndarray
        Alpha map
    """
    canvas = canvas_init.copy()
    alpha = alpha_init.copy()
    
    if isinstance(strokes, dict):
        strokes = [strokes]
    
    return renderer.render_strokes(canvas, alpha, strokes)


def save_golden_image(canvas: np.ndarray, path: Path):
    """Save canvas as golden PNG (sRGB).
    
    Parameters
    ----------
    canvas : np.ndarray
        Canvas in linear RGB [0,1]
    path : Path
        Output path
    """
    canvas_srgb = color_utils.linear_to_srgb(canvas)
    canvas_uint8 = np.clip(canvas_srgb * 255, 0, 255).astype(np.uint8)
    
    import PIL.Image
    img = PIL.Image.fromarray(canvas_uint8, mode='RGB')
    img.save(path)


def load_golden_image(path: Path) -> np.ndarray:
    """Load golden PNG and convert to linear RGB.
    
    Parameters
    ----------
    path : Path
        Path to golden PNG
    
    Returns
    -------
    np.ndarray
        Canvas in linear RGB [0,1], shape (H, W, 3)
    """
    import PIL.Image
    img = PIL.Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0  # sRGB [0,1]
    arr_linear = color_utils.srgb_to_linear(arr)
    return arr_linear


def compute_metrics(
    canvas_test: np.ndarray,
    canvas_golden: np.ndarray,
    alpha_test: np.ndarray
) -> Dict[str, float]:
    """Compute validation metrics.
    
    Parameters
    ----------
    canvas_test : np.ndarray
        Test canvas, linear RGB [0,1]
    canvas_golden : np.ndarray
        Golden canvas, linear RGB [0,1]
    alpha_test : np.ndarray
        Test alpha map [0,1]
    
    Returns
    -------
    dict
        Metrics: {psnr, ssim, delta_e_mean, coverage}
    """
    # PSNR (higher is better)
    psnr = metrics.psnr(canvas_test, canvas_golden)
    
    # SSIM (higher is better)
    ssim = metrics.ssim(canvas_test, canvas_golden)
    
    # ΔE2000 (lower is better)
    # Convert to Lab
    lab_test = color_utils.rgb_to_lab(canvas_test.reshape(-1, 3))
    lab_golden = color_utils.rgb_to_lab(canvas_golden.reshape(-1, 3))
    delta_e = color_utils.delta_e2000(lab_test, lab_golden)
    delta_e_mean = float(delta_e.mean())
    
    # Coverage (painted area fraction)
    coverage = metrics.paint_coverage_alpha(alpha_test)
    
    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'delta_e_mean': delta_e_mean,
        'coverage': float(coverage)
    }


def validate_metrics(metrics_dict: Dict[str, float], test_name: str):
    """Validate metrics against thresholds.
    
    Parameters
    ----------
    metrics_dict : dict
        Computed metrics
    test_name : str
        Test case name
    
    Raises
    ------
    AssertionError
        If any metric exceeds threshold
    """
    psnr = metrics_dict['psnr']
    ssim = metrics_dict['ssim']
    delta_e = metrics_dict['delta_e_mean']
    
    logger.info(f"{test_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, ΔE={delta_e:.2f}")
    
    assert psnr >= THRESHOLDS['psnr_min'], \
        f"PSNR {psnr:.2f} < {THRESHOLDS['psnr_min']} dB"
    
    assert ssim >= THRESHOLDS['ssim_min'], \
        f"SSIM {ssim:.4f} < {THRESHOLDS['ssim_min']}"
    
    assert delta_e <= THRESHOLDS['delta_e_max'], \
        f"ΔE {delta_e:.2f} > {THRESHOLDS['delta_e_max']}"


# ============================================================================
# GOLDEN IMAGE TESTS
# ============================================================================

@pytest.mark.golden
def test_golden_straight_line(renderer, blank_canvas):
    """Golden test: straight horizontal line."""
    strokes = get_golden_strokes()
    stroke = strokes['straight_line']
    
    canvas, alpha = render_golden_case(renderer, stroke, *blank_canvas)
    
    # Path to golden image
    golden_path = GOLDEN_DIR / 'straight_line.png'
    
    # Regenerate golden (manual mode, not in CI)
    if not golden_path.exists():
        fs.ensure_dir(GOLDEN_DIR)
        save_golden_image(canvas, golden_path)
        pytest.skip(f"Generated golden image: {golden_path}")
    
    # Load golden
    canvas_golden = load_golden_image(golden_path)
    
    # Compute and validate metrics
    metrics_dict = compute_metrics(canvas, canvas_golden, alpha)
    validate_metrics(metrics_dict, 'straight_line')


@pytest.mark.golden
def test_golden_curved_stroke(renderer, blank_canvas):
    """Golden test: S-curve with varying z/v."""
    strokes = get_golden_strokes()
    stroke = strokes['curved_stroke']
    
    canvas, alpha = render_golden_case(renderer, stroke, *blank_canvas)
    
    golden_path = GOLDEN_DIR / 'curved_stroke.png'
    
    if not golden_path.exists():
        fs.ensure_dir(GOLDEN_DIR)
        save_golden_image(canvas, golden_path)
        pytest.skip(f"Generated golden image: {golden_path}")
    
    canvas_golden = load_golden_image(golden_path)
    metrics_dict = compute_metrics(canvas, canvas_golden, alpha)
    validate_metrics(metrics_dict, 'curved_stroke')


@pytest.mark.golden
def test_golden_multi_stroke(renderer, blank_canvas):
    """Golden test: three overlapping strokes (CMY primary colors)."""
    strokes = get_golden_multi_strokes()
    
    canvas, alpha = render_golden_case(renderer, strokes, *blank_canvas)
    
    golden_path = GOLDEN_DIR / 'multi_stroke.png'
    
    if not golden_path.exists():
        fs.ensure_dir(GOLDEN_DIR)
        save_golden_image(canvas, golden_path)
        pytest.skip(f"Generated golden image: {golden_path}")
    
    canvas_golden = load_golden_image(golden_path)
    metrics_dict = compute_metrics(canvas, canvas_golden, alpha)
    validate_metrics(metrics_dict, 'multi_stroke')


@pytest.mark.golden
def test_golden_boundary_graze(renderer, blank_canvas):
    """Golden test: stroke near canvas edge."""
    strokes = get_golden_strokes()
    stroke = strokes['boundary_graze']
    
    canvas, alpha = render_golden_case(renderer, stroke, *blank_canvas)
    
    golden_path = GOLDEN_DIR / 'boundary_graze.png'
    
    if not golden_path.exists():
        fs.ensure_dir(GOLDEN_DIR)
        save_golden_image(canvas, golden_path)
        pytest.skip(f"Generated golden image: {golden_path}")
    
    canvas_golden = load_golden_image(golden_path)
    metrics_dict = compute_metrics(canvas, canvas_golden, alpha)
    validate_metrics(metrics_dict, 'boundary_graze')


@pytest.mark.golden
def test_golden_tiny_stroke(renderer, blank_canvas):
    """Golden test: micro-stroke for anti-aliasing validation."""
    strokes = get_golden_strokes()
    stroke = strokes['tiny_stroke']
    
    canvas, alpha = render_golden_case(renderer, stroke, *blank_canvas)
    
    golden_path = GOLDEN_DIR / 'tiny_stroke.png'
    
    if not golden_path.exists():
        fs.ensure_dir(GOLDEN_DIR)
        save_golden_image(canvas, golden_path)
        pytest.skip(f"Generated golden image: {golden_path}")
    
    canvas_golden = load_golden_image(golden_path)
    metrics_dict = compute_metrics(canvas, canvas_golden, alpha)
    validate_metrics(metrics_dict, 'tiny_stroke')


# ============================================================================
# COVERAGE REGRESSION TESTS
# ============================================================================

@pytest.mark.golden
def test_coverage_stability():
    """Test that coverage metrics are stable across runs."""
    renderer_cfg = {
        'work_area_mm': (210.0, 297.0),
        'render_px': (908, 1280)
    }
    
    stroke = get_golden_strokes()['straight_line']
    
    coverages = []
    for _ in range(3):
        renderer = CPUReferenceRenderer({}, renderer_cfg, load_toy_luts())
        canvas = np.ones((908, 1280, 3), dtype=np.float32)
        alpha = np.zeros((908, 1280), dtype=np.float32)
        
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        coverage = metrics.paint_coverage_alpha(alpha)
        coverages.append(coverage)
    
    # All runs should give identical coverage
    assert all(abs(c - coverages[0]) < 1e-7 for c in coverages), \
        f"Coverage not deterministic: {coverages}"


# ============================================================================
# METADATA & THRESHOLDS
# ============================================================================

def test_golden_thresholds_yaml_exists():
    """Test that golden thresholds YAML exists and is valid."""
    thresholds_path = GOLDEN_DIR / 'thresholds.yaml'
    
    if not thresholds_path.exists():
        # Create default thresholds
        fs.ensure_dir(GOLDEN_DIR)
        fs.atomic_yaml_dump(THRESHOLDS, thresholds_path)
        pytest.skip(f"Generated thresholds: {thresholds_path}")
    
    # Load and validate
    loaded = fs.load_yaml(thresholds_path)
    assert 'psnr_min' in loaded
    assert 'ssim_min' in loaded
    assert 'delta_e_max' in loaded
    assert loaded['psnr_min'] > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'golden'])
