"""GPU/CPU parity tests for reference renderer validation.

This test harness validates that the GPU (differentiable) renderer produces
results consistent with the CPU (reference) renderer within acceptable tolerances.

Test strategy:
1. Render same strokes with both CPU and GPU renderers
2. Compare outputs at render_px resolution
3. Validate metrics: PSNR, SSIM, ΔE2000, coverage difference
4. Check geometric properties: line width, endpoint positions

Parity thresholds:
- PSNR ≥ 26 dB (allow some GPU float16 precision loss)
- SSIM ≥ 0.90 (structural similarity)
- ΔE2000 mean ≤ 3.0 (perceptual difference)
- Coverage difference ≤ 0.02 (±2% relative)
- Line width difference ≤ 1 px (at FWHM)

Usage:
    pytest tests/test_parity_cpu_vs_gpu.py  # Run all parity tests
    pytest tests/test_parity_cpu_vs_gpu.py -k simple  # Run specific test
    pytest tests/test_parity_cpu_vs_gpu.py --skip-gpu  # Skip GPU tests (CI without CUDA)

Notes:
    - GPU tests are automatically skipped if CUDA is not available
    - When GPU renderer is not yet implemented, tests are marked xfail
    - Once GPU renderer is complete, remove xfail markers
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import color as color_utils, metrics


logger = logging.getLogger(__name__)

# Parity thresholds (looser than golden to account for GPU numerics)
PARITY_THRESHOLDS = {
    'psnr_min': 26.0,          # dB (allow some precision loss)
    'ssim_min': 0.90,          # [0,1]
    'delta_e_max': 3.0,        # ΔE2000
    'coverage_diff_max': 0.02,  # ±2% relative
    'width_diff_max': 1.0      # pixels
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def cpu_renderer():
    """Create CPU reference renderer."""
    env_cfg = {
        'work_area_mm': (210.0, 297.0),
        'render_px': (908, 1280)
    }
    sim_cfg = {}
    luts = load_toy_luts()
    return CPUReferenceRenderer(sim_cfg, env_cfg, luts)


@pytest.fixture
def gpu_renderer():
    """Create GPU differentiable renderer.
    
    Notes
    -----
    Currently returns None (not implemented). Tests will be skipped/xfail.
    When GPU renderer is ready, import and initialize it here.
    """
    # TODO: Import and initialize GPU renderer
    # from src.airbrush_simulator.differentiable_renderer import DifferentiableRenderer
    # return DifferentiableRenderer(...)
    return None


@pytest.fixture
def blank_canvas():
    """White canvas for rendering."""
    canvas = np.ones((908, 1280, 3), dtype=np.float32)
    alpha = np.zeros((908, 1280), dtype=np.float32)
    return canvas, alpha


@pytest.fixture
def simple_stroke():
    """Simple straight stroke for parity testing."""
    return {
        'id': 'parity-00000-simple',
        'bezier': {
            'p1': (50.0, 148.5),
            'p2': (80.0, 148.5),
            'p3': (130.0, 148.5),
            'p4': (160.0, 148.5)
        },
        'z_profile': {'z0': 10.0, 'z1': 10.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.8, 'm': 0.2, 'y': 0.1}
    }


@pytest.fixture
def curved_stroke():
    """Curved stroke for geometric parity."""
    return {
        'id': 'parity-00001-curved',
        'bezier': {
            'p1': (50.0, 50.0),
            'p2': (100.0, 120.0),
            'p3': (150.0, 80.0),
            'p4': (200.0, 150.0)
        },
        'z_profile': {'z0': 5.0, 'z1': 15.0},
        'speed_profile': {'v0': 100.0, 'v1': 200.0},
        'color_cmy': {'c': 0.2, 'm': 0.8, 'y': 0.3}
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_cpu(
    renderer: CPUReferenceRenderer,
    canvas: np.ndarray,
    alpha: np.ndarray,
    strokes: list
) -> Tuple[np.ndarray, np.ndarray]:
    """Render strokes with CPU renderer.
    
    Parameters
    ----------
    renderer : CPUReferenceRenderer
        CPU renderer
    canvas : np.ndarray
        Initial canvas
    alpha : np.ndarray
        Initial alpha map
    strokes : list of dict
        Stroke definitions
    
    Returns
    -------
    canvas : np.ndarray
        Rendered canvas (linear RGB)
    alpha : np.ndarray
        Alpha map
    """
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    if isinstance(strokes, dict):
        strokes = [strokes]
    
    return renderer.render_strokes(canvas, alpha, strokes)


def render_gpu(
    renderer,
    canvas: torch.Tensor,
    strokes: list
) -> torch.Tensor:
    """Render strokes with GPU renderer.
    
    Parameters
    ----------
    renderer
        GPU renderer (DifferentiableRenderer)
    canvas : torch.Tensor
        Initial canvas, shape (3, H, W), linear RGB
    strokes : list of dict
        Stroke definitions
    
    Returns
    -------
    torch.Tensor
        Rendered canvas (linear RGB)
    """
    # TODO: Implement GPU rendering path
    raise NotImplementedError("GPU renderer not yet implemented")


def compare_canvases(
    canvas_cpu: np.ndarray,
    canvas_gpu: np.ndarray,
    alpha_cpu: np.ndarray,
    alpha_gpu: np.ndarray
) -> Dict[str, float]:
    """Compare CPU and GPU rendered canvases.
    
    Parameters
    ----------
    canvas_cpu : np.ndarray
        CPU rendered canvas (linear RGB)
    canvas_gpu : np.ndarray
        GPU rendered canvas (linear RGB)
    alpha_cpu : np.ndarray
        CPU alpha map
    alpha_gpu : np.ndarray
        GPU alpha map
    
    Returns
    -------
    dict
        Comparison metrics: {psnr, ssim, delta_e_mean, coverage_diff}
    """
    # PSNR
    psnr = metrics.psnr(canvas_cpu, canvas_gpu)
    
    # SSIM
    ssim = metrics.ssim(canvas_cpu, canvas_gpu)
    
    # ΔE2000
    lab_cpu = color_utils.rgb_to_lab(canvas_cpu.reshape(-1, 3))
    lab_gpu = color_utils.rgb_to_lab(canvas_gpu.reshape(-1, 3))
    delta_e = color_utils.delta_e2000(lab_cpu, lab_gpu)
    delta_e_mean = float(delta_e.mean())
    
    # Coverage difference
    coverage_cpu = metrics.paint_coverage_alpha(alpha_cpu)
    coverage_gpu = metrics.paint_coverage_alpha(alpha_gpu)
    coverage_diff = abs(coverage_cpu - coverage_gpu)
    
    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'delta_e_mean': delta_e_mean,
        'coverage_diff': float(coverage_diff)
    }


def validate_parity(metrics_dict: Dict[str, float], test_name: str):
    """Validate parity metrics against thresholds.
    
    Parameters
    ----------
    metrics_dict : dict
        Computed comparison metrics
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
    coverage_diff = metrics_dict['coverage_diff']
    
    logger.info(
        f"{test_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
        f"ΔE={delta_e:.2f}, Δcoverage={coverage_diff:.4f}"
    )
    
    assert psnr >= PARITY_THRESHOLDS['psnr_min'], \
        f"PSNR {psnr:.2f} < {PARITY_THRESHOLDS['psnr_min']} dB"
    
    assert ssim >= PARITY_THRESHOLDS['ssim_min'], \
        f"SSIM {ssim:.4f} < {PARITY_THRESHOLDS['ssim_min']}"
    
    assert delta_e <= PARITY_THRESHOLDS['delta_e_max'], \
        f"ΔE {delta_e:.2f} > {PARITY_THRESHOLDS['delta_e_max']}"
    
    assert coverage_diff <= PARITY_THRESHOLDS['coverage_diff_max'], \
        f"Coverage diff {coverage_diff:.4f} > {PARITY_THRESHOLDS['coverage_diff_max']}"


# ============================================================================
# PARITY TESTS
# ============================================================================

@pytest.mark.parity
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(reason="GPU renderer not yet implemented", strict=False)
def test_parity_simple_stroke(cpu_renderer, gpu_renderer, blank_canvas, simple_stroke):
    """Parity test: simple straight stroke."""
    if gpu_renderer is None:
        pytest.skip("GPU renderer not available")
    
    # Render with CPU
    canvas_cpu, alpha_cpu = render_cpu(
        cpu_renderer, *blank_canvas, simple_stroke
    )
    
    # Render with GPU
    canvas_gpu_np = np.ones((908, 1280, 3), dtype=np.float32)
    canvas_gpu = torch.from_numpy(canvas_gpu_np).permute(2, 0, 1).cuda()
    canvas_gpu = render_gpu(gpu_renderer, canvas_gpu, simple_stroke)
    canvas_gpu_np = canvas_gpu.permute(1, 2, 0).cpu().numpy()
    
    # TODO: Extract alpha from GPU renderer
    alpha_gpu = np.zeros((908, 1280), dtype=np.float32)
    
    # Compare
    metrics_dict = compare_canvases(canvas_cpu, canvas_gpu_np, alpha_cpu, alpha_gpu)
    validate_parity(metrics_dict, 'simple_stroke')


@pytest.mark.parity
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(reason="GPU renderer not yet implemented", strict=False)
def test_parity_curved_stroke(cpu_renderer, gpu_renderer, blank_canvas, curved_stroke):
    """Parity test: curved stroke with varying z/v."""
    if gpu_renderer is None:
        pytest.skip("GPU renderer not available")
    
    # Render with CPU
    canvas_cpu, alpha_cpu = render_cpu(
        cpu_renderer, *blank_canvas, curved_stroke
    )
    
    # Render with GPU
    canvas_gpu_np = np.ones((908, 1280, 3), dtype=np.float32)
    canvas_gpu = torch.from_numpy(canvas_gpu_np).permute(2, 0, 1).cuda()
    canvas_gpu = render_gpu(gpu_renderer, canvas_gpu, curved_stroke)
    canvas_gpu_np = canvas_gpu.permute(1, 2, 0).cpu().numpy()
    
    # TODO: Extract alpha from GPU renderer
    alpha_gpu = np.zeros((908, 1280), dtype=np.float32)
    
    # Compare
    metrics_dict = compare_canvases(canvas_cpu, canvas_gpu_np, alpha_cpu, alpha_gpu)
    validate_parity(metrics_dict, 'curved_stroke')


@pytest.mark.parity
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(reason="GPU renderer not yet implemented", strict=False)
def test_parity_multi_stroke(cpu_renderer, gpu_renderer, blank_canvas):
    """Parity test: three overlapping strokes."""
    if gpu_renderer is None:
        pytest.skip("GPU renderer not available")
    
    # Three strokes
    strokes = [
        {
            'id': 'parity-multi-00',
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
            'id': 'parity-multi-01',
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
            'id': 'parity-multi-02',
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
    
    # Render with CPU
    canvas_cpu, alpha_cpu = render_cpu(cpu_renderer, *blank_canvas, strokes)
    
    # Render with GPU
    canvas_gpu_np = np.ones((908, 1280, 3), dtype=np.float32)
    canvas_gpu = torch.from_numpy(canvas_gpu_np).permute(2, 0, 1).cuda()
    canvas_gpu = render_gpu(gpu_renderer, canvas_gpu, strokes)
    canvas_gpu_np = canvas_gpu.permute(1, 2, 0).cpu().numpy()
    
    # TODO: Extract alpha from GPU renderer
    alpha_gpu = np.zeros((908, 1280), dtype=np.float32)
    
    # Compare
    metrics_dict = compare_canvases(canvas_cpu, canvas_gpu_np, alpha_cpu, alpha_gpu)
    validate_parity(metrics_dict, 'multi_stroke')


# ============================================================================
# GEOMETRIC PARITY TESTS
# ============================================================================

@pytest.mark.parity
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(reason="GPU renderer not yet implemented", strict=False)
def test_parity_line_width(cpu_renderer, gpu_renderer, blank_canvas, simple_stroke):
    """Test that line width (FWHM) matches between CPU and GPU."""
    if gpu_renderer is None:
        pytest.skip("GPU renderer not available")
    
    # Render with CPU
    _, alpha_cpu = render_cpu(cpu_renderer, *blank_canvas, simple_stroke)
    
    # Render with GPU
    # TODO: Get alpha from GPU renderer
    alpha_gpu = np.zeros((908, 1280), dtype=np.float32)
    
    # Measure FWHM at mid-stroke
    h = alpha_cpu.shape[0]
    cy = h // 2
    
    profile_cpu = alpha_cpu[cy, :]
    profile_gpu = alpha_gpu[cy, :]
    
    # Find FWHM for both
    def measure_fwhm(profile):
        peak = profile.max()
        if peak < 0.1:
            return 0.0
        half_max = 0.5 * peak
        above = profile >= half_max
        indices = np.where(above)[0]
        if len(indices) < 2:
            return 0.0
        return float(indices[-1] - indices[0])
    
    fwhm_cpu = measure_fwhm(profile_cpu)
    fwhm_gpu = measure_fwhm(profile_gpu)
    
    width_diff = abs(fwhm_cpu - fwhm_gpu)
    
    logger.info(f"FWHM: CPU={fwhm_cpu:.2f} px, GPU={fwhm_gpu:.2f} px, diff={width_diff:.2f} px")
    
    assert width_diff <= PARITY_THRESHOLDS['width_diff_max'], \
        f"Width difference {width_diff:.2f} > {PARITY_THRESHOLDS['width_diff_max']} px"


# ============================================================================
# METADATA
# ============================================================================

def test_parity_thresholds_documented():
    """Test that parity thresholds are documented and reasonable."""
    assert PARITY_THRESHOLDS['psnr_min'] > 0
    assert PARITY_THRESHOLDS['ssim_min'] > 0
    assert PARITY_THRESHOLDS['delta_e_max'] > 0
    assert PARITY_THRESHOLDS['coverage_diff_max'] > 0
    
    # Sanity check ranges
    assert 20 <= PARITY_THRESHOLDS['psnr_min'] <= 40
    assert 0.8 <= PARITY_THRESHOLDS['ssim_min'] <= 1.0
    assert 1.0 <= PARITY_THRESHOLDS['delta_e_max'] <= 10.0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'parity'])
