"""Unit tests for pen tracer module.

Tests:
    - Black mask extraction from LAB thresholds
    - Morphological cleanup
    - Component classification (line vs region)
    - Contour packing around centerlines
    - Region hatching with overshading
    - Path ordering (nearest neighbor)
    - Preview rendering
    - Metrics computation
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import pytest

from src.data_pipeline import pen_tracer
from src.utils import validators, color


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
        offset_mm=[0.0, 0.0, 0.0],
        macros=validators.PenMacros(pen_up="PEN_UP.gcode", pen_down="PEN_DOWN.gcode"),
        accel=validators.PenAcceleration(xy_mm_s2=1500.0),
        joins=validators.PenJoins(cap_style="round", join_style="round"),
    )


@pytest.fixture
def simple_pen_tracer_config():
    """Simple pen tracer configuration for testing."""
    return validators.PenTracerV1(
        schema="pen_tracer.v1",
        thresholds=validators.PenThresholds(lab_l_max=18.0, a_abs_min=0.0, b_abs_min=0.0),
        morphology=validators.PenMorphology(close_px=1, open_px=0, min_area_px=16),
        vectorization=validators.PenVectorization(simplify_tol_px=1.2, potrace_turdsize=2),
        classification=validators.PenClassification(
            line_like_width_px=6, donut_hole_min_area_px=36
        ),
        filling=validators.PenFilling(
            hatch_angles_deg=[0.0],
            hatch_spacing_scale=1.0,
            darkness_to_passes=[
                validators.DarknessToPassesRule(l_max=14.0, passes=3),
                validators.DarknessToPassesRule(l_max=16.0, passes=2),
                validators.DarknessToPassesRule(l_max=18.0, passes=1),
            ],
        ),
        contours=validators.PenContours(endcap_extra_len_mm=0.3, max_shells_per_side=12),
        visibility=validators.PenVisibility(min_coverage=0.0005, max_gap_frac=0.02),
        debug=validators.PenDebug(save_intermediates=False),
    )


def test_extract_black_mask_simple(simple_pen_tracer_config):
    """Test black mask extraction from simple synthetic image."""
    # Create synthetic image: white background with black line
    H, W = 100, 100
    img_linear = torch.ones((3, H, W), dtype=torch.float32)  # White
    
    # Add black horizontal line
    img_linear[:, 45:55, 20:80] = 0.0  # Pure black
    
    # Extract mask
    mask = pen_tracer.extract_black_mask(img_linear, simple_pen_tracer_config)
    
    # Check shape and dtype
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8
    
    # Check coverage (should be approximately 10 * 60 / 10000 = 0.06)
    coverage = np.mean(mask > 0)
    assert 0.05 < coverage < 0.07, f"Coverage {coverage} out of expected range"


def test_morphological_cleanup(simple_pen_tracer_config):
    """Test morphological operations remove noise."""
    # Create noisy mask
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Add large component
    mask[30:70, 30:70] = 255
    
    # Add small noise components
    mask[10:12, 10:12] = 255  # 4 pixels (should be removed)
    mask[90:92, 90:92] = 255  # 4 pixels (should be removed)
    
    # Clean
    cleaned = pen_tracer.morphological_cleanup(mask, simple_pen_tracer_config)
    
    # Check noise removed (only large component should remain)
    # Large component area: 40*40 = 1600 pixels
    assert np.sum(cleaned > 0) > 1500
    assert np.sum(cleaned > 0) < 1700  # Allow some morphology changes


def test_classify_components_line_vs_region(simple_pen_tracer_config):
    """Test component classification based on width."""
    H, W = 200, 200
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Add thin line (width ~4 px, should be line-like)
    mask[50:54, 50:150] = 255
    
    # Add thick region (width ~40 px, should be region)
    mask[100:140, 50:150] = 255
    
    # Classify
    line_masks, region_masks, dt = pen_tracer.classify_components(
        mask, simple_pen_tracer_config
    )
    
    # Should have 1 line and 1 region
    assert len(line_masks) == 1, f"Expected 1 line, got {len(line_masks)}"
    assert len(region_masks) == 1, f"Expected 1 region, got {len(region_masks)}"


def test_pack_contours_generates_shells(
    simple_pen_tool_config, simple_pen_tracer_config
):
    """Test contour packing generates multiple shells."""
    # Simple centerline
    centerline_mm = np.array([
        [10.0, 10.0],
        [20.0, 10.0],
        [30.0, 10.0],
    ], dtype=np.float32)
    
    # Constant width (5mm half-width)
    local_width_mm = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    
    # Pack contours
    shells = pen_tracer.pack_contours_around_centerline(
        centerline_mm,
        local_width_mm,
        simple_pen_tool_config,
        simple_pen_tracer_config,
    )
    
    # Should generate multiple shells (center + offsets on both sides)
    # tip = 0.3mm, overlap = 0.25 → step = 0.225mm
    # max_half_width = 5mm → k_max = floor(5/0.225) = 22 shells per side
    # Total: 1 (center) + 2*min(22, max_shells_per_side=12) = 25
    assert len(shells) > 10, f"Expected >10 shells, got {len(shells)}"


def test_render_pen_preview_basic(simple_pen_tool_config):
    """Test pen preview rendering produces reasonable output."""
    # Simple paths
    paths = [
        np.array([[10.0, 10.0], [50.0, 10.0]], dtype=np.float32),
        np.array([[10.0, 20.0], [50.0, 20.0]], dtype=np.float32),
    ]
    
    render_px = (100, 100)
    mm_per_px = 1.0  # 1mm = 1px
    
    # Render
    preview = pen_tracer.render_pen_preview(
        paths,
        render_px,
        simple_pen_tool_config.tip_diameter_mm,
        mm_per_px,
    )
    
    # Check shape
    assert preview.shape == (100, 100)
    assert preview.dtype == np.uint8
    
    # Check not all white (should have black lines)
    assert np.mean(preview == 255) < 0.99, "Preview should not be all white"
    
    # Check not all black
    assert np.mean(preview == 0) < 0.5, "Preview should not be mostly black"


def test_order_paths_nearest_neighbor():
    """Test path ordering reduces travel distance."""
    # Create paths in scattered order
    paths = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),  # Near origin
        np.array([[100.0, 100.0], [110.0, 100.0]]),  # Far
        np.array([[10.0, 5.0], [20.0, 5.0]]),  # Near first path
    ]
    
    # Order
    ordered = pen_tracer.order_paths_nearest_neighbor(paths, start_pos=np.array([0.0, 0.0]))
    
    # First path should be the one nearest to origin
    assert np.allclose(ordered[0][0], [0.0, 0.0], atol=0.1)
    
    # Last path should be the far one
    assert np.allclose(ordered[-1][0][:2], [100.0, 100.0], atol=0.1) or \
           np.allclose(ordered[-1][0][:2], [110.0, 100.0], atol=0.1)


def test_compute_pen_metrics_coverage(simple_pen_tracer_config):
    """Test metrics computation returns reasonable values."""
    # Create preview with known coverage
    preview = np.ones((100, 100), dtype=np.uint8) * 255  # White
    preview[40:60, 40:60] = 0  # 20x20 black square = 400/10000 = 0.04
    
    # No region masks for this test
    region_masks = []
    
    # Compute metrics
    metrics_dict = pen_tracer.compute_pen_metrics(
        preview, region_masks, simple_pen_tracer_config
    )
    
    # Check coverage
    assert 'coverage_black' in metrics_dict
    assert 0.03 < metrics_dict['coverage_black'] < 0.05, \
        f"Coverage {metrics_dict['coverage_black']} out of expected range"
    
    # Check luminance
    assert 'mean_luminance' in metrics_dict
    assert 0.95 < metrics_dict['mean_luminance'] < 0.97


def test_extend_polyline_endcaps():
    """Test endcap extension produces correct length."""
    polyline = np.array([
        [10.0, 10.0],
        [20.0, 10.0],
        [30.0, 10.0],
    ], dtype=np.float32)
    
    extra_len = 2.0
    
    extended = pen_tracer.extend_polyline_endcaps(polyline, extra_len)
    
    # Should have 2 extra points
    assert len(extended) == len(polyline) + 2
    
    # Check start extension (should be before first point)
    assert extended[0][0] < polyline[0][0] - 1.5  # Extended in -X direction


def test_vectorize_line_component_simple(simple_pen_tracer_config):
    """Test line vectorization produces reasonable output."""
    # Create simple line mask
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[45:55, 50:150] = 255  # 10px wide horizontal line
    
    # Distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    mm_per_px = 0.5
    
    # Vectorize
    centerline_mm, local_width_mm = pen_tracer.vectorize_line_component(
        mask, dist_transform, simple_pen_tracer_config, mm_per_px
    )
    
    # Should produce points
    assert len(centerline_mm) > 0, "Should produce centerline points"
    assert len(local_width_mm) == len(centerline_mm), "Width array should match centerline"
    
    # Check width is reasonable (should be ~5px * 0.5 mm/px = 2.5mm)
    assert np.mean(local_width_mm) > 1.5
    assert np.mean(local_width_mm) < 3.5


@pytest.mark.parametrize("coverage,expected_pass", [
    (0.001, True),   # Above min_coverage
    (0.0001, False),  # Below min_coverage
])
def test_visibility_gate(coverage, expected_pass, simple_pen_tracer_config):
    """Test visibility gate correctly filters low-coverage layers."""
    # Create preview with specified coverage
    H, W = 1000, 1000
    preview = np.ones((H, W), dtype=np.uint8) * 255
    
    # Set coverage by painting black pixels
    num_black = int(coverage * H * W)
    preview.flat[:num_black] = 0
    
    metrics_dict = pen_tracer.compute_pen_metrics(
        preview, [], simple_pen_tracer_config
    )
    
    if expected_pass:
        assert metrics_dict['coverage_black'] >= simple_pen_tracer_config.visibility.min_coverage
    else:
        assert metrics_dict['coverage_black'] < simple_pen_tracer_config.visibility.min_coverage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

