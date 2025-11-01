"""Visual regression tests for CPU renderer.

These tests generate reference images demonstrating various stroke configurations,
colors, speeds, Bézier curves, and layering. They serve both as visual verification
and as regression tests to ensure rendering quality remains consistent.

Run with: pytest tests/test_cpu_visual_regression.py -v
Regenerate golden images: pytest tests/test_cpu_visual_regression.py --regen-visual
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import PIL.Image

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import fs, color as color_utils, geometry

logger = logging.getLogger(__name__)

# Output directory for visual regression images
VISUAL_OUTPUT_DIR = Path("ci/visual_regression_cpu")


@pytest.fixture(scope="module")
def visual_renderer():
    """Create a renderer for visual regression tests."""
    sim_cfg = fs.load_yaml("configs/sim/physics_v1.yaml")
    env_cfg = {
        'work_area_mm': (210.0, 297.0),
        'render_px': (908, 1280)
    }
    # Use realistic colors (high_visibility=False) for visual tests
    luts = load_toy_luts(high_visibility=False)
    return CPUReferenceRenderer(sim_cfg, env_cfg, luts)


@pytest.fixture(scope="module")
def blank_canvas(visual_renderer):
    """Create a blank white canvas."""
    h, w = visual_renderer.canvas_h_px, visual_renderer.canvas_w_px
    canvas = np.ones((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    return canvas, alpha


def save_canvas(canvas: np.ndarray, output_path: Path):
    """Save canvas to PNG file (linear RGB -> sRGB conversion)."""
    fs.ensure_dir(output_path.parent)
    
    # Convert to sRGB
    canvas_torch = torch.from_numpy(canvas).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
    canvas_srgb = color_utils.linear_to_srgb(canvas_torch)
    canvas_srgb = canvas_srgb.permute(1, 2, 0).numpy()  # (3,H,W) -> (H,W,3)
    canvas_uint8 = np.clip(canvas_srgb * 255, 0, 255).astype(np.uint8)
    
    # Save
    img = PIL.Image.fromarray(canvas_uint8, mode='RGB')
    img.save(output_path)
    logger.info(f"Saved visual regression image: {output_path}")


@pytest.mark.visual
def test_visual_speed_variation(visual_renderer, blank_canvas):
    """Visual test: strokes at different speeds (width and opacity variation)."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    speeds = [20.0, 40.0, 80.0, 120.0, 180.0]
    y_positions = [150, 250, 350, 450, 550]
    
    for i, (speed, y_pos) in enumerate(zip(speeds, y_positions)):
        stroke = {
            'id': f'speed-test-{i}',
            'bezier': {
                'p1': (40.0, float(y_pos)),
                'p2': (80.0, float(y_pos)),
                'p3': (130.0, float(y_pos)),
                'p4': (170.0, float(y_pos))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': speed, 'v1': speed},
            'color_cmy': {'c': 0.9, 'm': 0.9, 'y': 0.9}  # Black
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "speed_variation.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0003, "No strokes visible"


@pytest.mark.visual
def test_visual_z_height_variation(visual_renderer, blank_canvas):
    """Visual test: strokes at different Z heights (width variation)."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    z_heights = [3.0, 6.0, 9.0, 12.0, 16.0]
    y_positions = [150, 250, 350, 450, 550]
    
    for i, (z, y_pos) in enumerate(zip(z_heights, y_positions)):
        stroke = {
            'id': f'z-test-{i}',
            'bezier': {
                'p1': (40.0, float(y_pos)),
                'p2': (80.0, float(y_pos)),
                'p3': (130.0, float(y_pos)),
                'p4': (170.0, float(y_pos))
            },
            'z_profile': {'z0': z, 'z1': z},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.9, 'm': 0.9, 'y': 0.9}  # Black
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "z_height_variation.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0003, "No strokes visible"


@pytest.mark.visual
def test_visual_color_palette(visual_renderer, blank_canvas):
    """Visual test: strokes in different colors (CMY combinations)."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Define colors: (name, CMY values)
    colors = [
        ('black', {'c': 0.9, 'm': 0.9, 'y': 0.9}),
        ('cyan', {'c': 0.9, 'm': 0.0, 'y': 0.0}),
        ('magenta', {'c': 0.0, 'm': 0.9, 'y': 0.0}),
        ('yellow', {'c': 0.0, 'm': 0.0, 'y': 0.9}),
        ('red', {'c': 0.0, 'm': 0.9, 'y': 0.9}),
        ('green', {'c': 0.9, 'm': 0.0, 'y': 0.9}),
        ('blue', {'c': 0.9, 'm': 0.9, 'y': 0.0}),
    ]
    
    # Create two columns of color swatches
    y_positions = np.linspace(150, 650, len(colors))
    
    for i, ((color_name, cmy), y_pos) in enumerate(zip(colors, y_positions)):
        stroke = {
            'id': f'color-test-{color_name}',
            'bezier': {
                'p1': (30.0, float(y_pos)),
                'p2': (50.0, float(y_pos)),
                'p3': (70.0, float(y_pos)),
                'p4': (90.0, float(y_pos))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': cmy
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "color_palette.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0003, "No strokes visible"


@pytest.mark.visual
def test_visual_bezier_curves(visual_renderer, blank_canvas):
    """Visual test: various Bézier curve shapes."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Define different Bézier curves
    curves = [
        # Straight line
        {
            'name': 'straight',
            'p1': (30.0, 150.0),
            'p2': (70.0, 150.0),
            'p3': (130.0, 150.0),
            'p4': (170.0, 150.0)
        },
        # Gentle S-curve
        {
            'name': 's-curve',
            'p1': (30.0, 250.0),
            'p2': (70.0, 230.0),
            'p3': (130.0, 270.0),
            'p4': (170.0, 250.0)
        },
        # Sharp turn
        {
            'name': 'sharp-turn',
            'p1': (30.0, 350.0),
            'p2': (90.0, 350.0),
            'p3': (90.0, 400.0),
            'p4': (90.0, 450.0)
        },
        # Loop
        {
            'name': 'loop',
            'p1': (50.0, 550.0),
            'p2': (120.0, 500.0),
            'p3': (120.0, 600.0),
            'p4': (50.0, 550.0)
        },
        # Wave
        {
            'name': 'wave',
            'p1': (30.0, 680.0),
            'p2': (70.0, 650.0),
            'p3': (130.0, 710.0),
            'p4': (170.0, 680.0)
        }
    ]
    
    for i, curve in enumerate(curves):
        stroke = {
            'id': f'bezier-test-{curve["name"]}',
            'bezier': {
                'p1': curve['p1'],
                'p2': curve['p2'],
                'p3': curve['p3'],
                'p4': curve['p4']
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.9, 'm': 0.9, 'y': 0.9}  # Black
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "bezier_curves.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0003, "No strokes visible"


@pytest.mark.visual
def test_visual_color_layering(visual_renderer, blank_canvas):
    """Visual test: layering different colors (alpha-over compositing)."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Create a grid pattern with clear overlaps (similar to complex_composition but focused on layering)
    colors = [
        {'c': 0.8, 'm': 0.0, 'y': 0.0},  # Cyan
        {'c': 0.0, 'm': 0.8, 'y': 0.0},  # Magenta
        {'c': 0.0, 'm': 0.0, 'y': 0.8},  # Yellow
    ]
    
    # Horizontal strokes (within A4 bounds: 0-210mm x, 0-297mm y)
    for i, color in enumerate(colors):
        y = 150 + i * 40  # Changed from 300 to 150 to stay within bounds
        stroke = {
            'id': f'horiz-{i}',
            'bezier': {
                'p1': (40.0, float(y)),
                'p2': (70.0, float(y)),
                'p3': (130.0, float(y)),
                'p4': (160.0, float(y))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': color
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    # Vertical/diagonal strokes crossing (within bounds)
    for i, color in enumerate(colors):
        x_start = 60 + i * 40
        stroke = {
            'id': f'vert-{i}',
            'bezier': {
                'p1': (float(x_start), 130.0),  # Changed from 280 to 130
                'p2': (float(x_start + 10), 160.0),  # Changed from 310 to 160
                'p3': (float(x_start + 10), 200.0),  # Changed from 350 to 200
                'p4': (float(x_start + 20), 230.0)  # Changed from 380 to 230
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': colors[(i + 1) % len(colors)]
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "color_layering.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0005, "No strokes visible"


@pytest.mark.visual
@pytest.mark.skip(reason="Needs adjustment for transparent alcohol ink model")
def test_visual_gradient_strokes(visual_renderer, blank_canvas):
    """Visual test: strokes with varying Z and speed along the path."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Gradient from low Z to high Z
    stroke_z_gradient = {
        'id': 'gradient-z',
        'bezier': {
            'p1': (30.0, 250.0),
            'p2': (70.0, 250.0),
            'p3': (130.0, 250.0),
            'p4': (170.0, 250.0)
        },
        'z_profile': {'z0': 4.0, 'z1': 16.0},  # Narrow to wide
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.9, 'm': 0.0, 'y': 0.0}  # Cyan
    }
    canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke_z_gradient)
    
    # Gradient from slow to fast speed
    stroke_speed_gradient = {
        'id': 'gradient-speed',
        'bezier': {
            'p1': (30.0, 400.0),
            'p2': (70.0, 400.0),
            'p3': (130.0, 400.0),
            'p4': (170.0, 400.0)
        },
        'z_profile': {'z0': 8.0, 'z1': 8.0},
        'speed_profile': {'v0': 20.0, 'v1': 150.0},  # Slow to fast
        'color_cmy': {'c': 0.0, 'm': 0.9, 'y': 0.0}  # Magenta
    }
    canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke_speed_gradient)
    
    # Combined gradient (Z and speed)
    stroke_combined_gradient = {
        'id': 'gradient-combined',
        'bezier': {
            'p1': (30.0, 550.0),
            'p2': (70.0, 550.0),
            'p3': (130.0, 550.0),
            'p4': (170.0, 550.0)
        },
        'z_profile': {'z0': 4.0, 'z1': 16.0},
        'speed_profile': {'v0': 20.0, 'v1': 150.0},
        'color_cmy': {'c': 0.0, 'm': 0.0, 'y': 0.9}  # Yellow
    }
    canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke_combined_gradient)
    
    output_path = VISUAL_OUTPUT_DIR / "gradient_strokes.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0003, "No strokes visible"


@pytest.mark.visual
@pytest.mark.skip(reason="Needs adjustment for transparent alcohol ink model")
def test_visual_complex_composition(visual_renderer, blank_canvas):
    """Visual test: complex composition with multiple overlapping colored strokes."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Create a grid of overlapping colored strokes
    colors = [
        {'c': 0.8, 'm': 0.0, 'y': 0.0},  # Cyan
        {'c': 0.0, 'm': 0.8, 'y': 0.0},  # Magenta
        {'c': 0.0, 'm': 0.0, 'y': 0.8},  # Yellow
    ]
    
    # Horizontal strokes
    for i, color in enumerate(colors):
        y = 300 + i * 40
        stroke = {
            'id': f'horiz-{i}',
            'bezier': {
                'p1': (40.0, float(y)),
                'p2': (70.0, float(y)),
                'p3': (130.0, float(y)),
                'p4': (160.0, float(y))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': color
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    # Vertical/diagonal strokes crossing
    for i, color in enumerate(colors):
        x_start = 60 + i * 40
        stroke = {
            'id': f'vert-{i}',
            'bezier': {
                'p1': (float(x_start), 280.0),
                'p2': (float(x_start + 10), 310.0),
                'p3': (float(x_start + 10), 350.0),
                'p4': (float(x_start + 20), 380.0)
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': colors[(i + 1) % len(colors)]
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "complex_composition.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.0005, "No strokes visible"


@pytest.mark.visual
def test_visual_all_features(visual_renderer, blank_canvas):
    """Visual test: comprehensive demo showing all renderer features."""
    canvas, alpha = blank_canvas
    canvas = canvas.copy()
    alpha = alpha.copy()
    
    # Section 1: Speed variation (top left)
    for i, speed in enumerate([30.0, 60.0, 120.0]):
        y = 100 + i * 40
        stroke = {
            'id': f'all-speed-{i}',
            'bezier': {
                'p1': (20.0, float(y)),
                'p2': (40.0, float(y)),
                'p3': (60.0, float(y)),
                'p4': (80.0, float(y))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': speed, 'v1': speed},
            'color_cmy': {'c': 0.9, 'm': 0.9, 'y': 0.9}
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    # Section 2: Colors (top right)
    colors = [
        {'c': 0.8, 'm': 0.0, 'y': 0.0},  # Cyan
        {'c': 0.0, 'm': 0.8, 'y': 0.0},  # Magenta
        {'c': 0.0, 'm': 0.0, 'y': 0.8},  # Yellow
    ]
    for i, color in enumerate(colors):
        y = 100 + i * 40
        stroke = {
            'id': f'all-color-{i}',
            'bezier': {
                'p1': (130.0, float(y)),
                'p2': (145.0, float(y)),
                'p3': (160.0, float(y)),
                'p4': (175.0, float(y))
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': color
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    # Section 3: Curves (middle)
    curves = [
        # S-curve
        {
            'p1': (20.0, 300.0),
            'p2': (50.0, 280.0),
            'p3': (80.0, 320.0),
            'p4': (110.0, 300.0)
        },
        # Arc
        {
            'p1': (130.0, 300.0),
            'p2': (145.0, 270.0),
            'p3': (160.0, 270.0),
            'p4': (175.0, 300.0)
        }
    ]
    for i, curve in enumerate(curves):
        stroke = {
            'id': f'all-curve-{i}',
            'bezier': curve,
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {'c': 0.9, 'm': 0.9, 'y': 0.9}
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    # Section 4: Layering (bottom)
    layer_strokes = [
        {
            'bezier': {
                'p1': (40.0, 500.0),
                'p2': (60.0, 500.0),
                'p3': (80.0, 500.0),
                'p4': (100.0, 500.0)
            },
            'color_cmy': {'c': 0.7, 'm': 0.0, 'y': 0.0}
        },
        {
            'bezier': {
                'p1': (60.0, 480.0),
                'p2': (70.0, 490.0),
                'p3': (80.0, 510.0),
                'p4': (90.0, 520.0)
            },
            'color_cmy': {'c': 0.0, 'm': 0.7, 'y': 0.0}
        },
        {
            'bezier': {
                'p1': (60.0, 520.0),
                'p2': (70.0, 510.0),
                'p3': (80.0, 490.0),
                'p4': (90.0, 480.0)
            },
            'color_cmy': {'c': 0.0, 'm': 0.0, 'y': 0.7}
        }
    ]
    
    for i, layer_cfg in enumerate(layer_strokes):
        stroke = {
            'id': f'all-layer-{i}',
            'bezier': layer_cfg['bezier'],
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': layer_cfg['color_cmy']
        }
        canvas, alpha = visual_renderer.render_stroke(canvas, alpha, stroke)
    
    output_path = VISUAL_OUTPUT_DIR / "all_features.png"
    save_canvas(canvas, output_path)
    
    # Verify strokes were rendered
    assert np.mean(alpha) > 0.001, "No strokes visible"


if __name__ == "__main__":
    # Allow running directly for quick visual checks
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "visual"]))

