#!/usr/bin/env python3
"""Comprehensive alcohol ink renderer demo.

Demonstrates all renderer capabilities in a single image:
- Speed variation (fast=light, slow=dark)
- Z-height variation (low=narrow, high=wide)
- Color variety (CMY spectrum)
- Transparent layering (overlapping strokes show blending)
- Bézier curves
- Within A4 canvas bounds (0-210mm x, 0-297mm y)
"""
import sys
from pathlib import Path
import numpy as np
import PIL.Image
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import fs, color as color_utils

def main():
    """Create comprehensive demo image."""
    print("Loading renderer configuration...")
    sim_cfg = fs.load_yaml("configs/sim/physics_v1.yaml")
    env_cfg = {
        'work_area_mm': (210.0, 297.0),
        'render_px': (1280, 908)  # A4 landscape
    }
    
    # Use realistic colors (not high_visibility black)
    luts = load_toy_luts(high_visibility=False)
    renderer = CPUReferenceRenderer(sim_cfg, env_cfg, luts)
    
    # Initialize blank canvas
    H, W = env_cfg['render_px']
    canvas = np.ones((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    
    print("Rendering comprehensive demo...")
    
    # Define color palette (7 distinct colors in CMY space - SATURATED)
    colors = [
        {'c': 0.9, 'm': 0.0, 'y': 0.0, 'name': 'cyan'},
        {'c': 0.0, 'm': 0.9, 'y': 0.0, 'name': 'magenta'},
        {'c': 0.0, 'm': 0.0, 'y': 0.9, 'name': 'yellow'},
        {'c': 0.0, 'm': 0.9, 'y': 0.9, 'name': 'red'},
        {'c': 0.9, 'm': 0.0, 'y': 0.9, 'name': 'green'},
        {'c': 0.9, 'm': 0.9, 'y': 0.0, 'name': 'blue'},
        {'c': 0.6, 'm': 0.3, 'y': 0.8, 'name': 'orange'},
    ]
    
    # =========================================================================
    # Section 1: Within-Stroke Speed Gradients (horizontal strokes, top section)
    # Each stroke accelerates: slow start → fast end = dark → light
    # =========================================================================
    print("  Section 1: Within-stroke speed gradients (accelerating)...")
    speed_pairs = [(25.0, 180.0), (30.0, 150.0), (40.0, 120.0), (50.0, 100.0), (60.0, 80.0)]
    y_start = 30.0
    y_spacing = 25.0
    
    for i, (v0, v1) in enumerate(speed_pairs):
        color = colors[i % len(colors)]
        y_pos = y_start + i * y_spacing
        
        stroke = {
            'id': f'speed-gradient-{i}',
            'bezier': {
                'p1': (20.0, y_pos),
                'p2': (70.0, y_pos),
                'p3': (140.0, y_pos),
                'p4': (190.0, y_pos)
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': v0, 'v1': v1},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    {color['name']:8s} @ {v0:5.1f}→{v1:5.1f} mm/s")
    
    # =========================================================================
    # Section 2: Within-Stroke Z-Height Gradients (horizontal strokes, middle section)
    # Each stroke lifts: low start → high end = narrow → wide
    # =========================================================================
    print("  Section 2: Within-stroke Z-height gradients (lifting)...")
    z_pairs = [(3.0, 18.0), (4.0, 15.0), (5.0, 12.0), (6.0, 10.0), (7.0, 9.0)]
    y_start = 160.0
    y_spacing = 25.0
    
    for i, (z0, z1) in enumerate(z_pairs):
        color = colors[(i + 2) % len(colors)]
        y_pos = y_start + i * y_spacing
        
        stroke = {
            'id': f'z-gradient-{i}',
            'bezier': {
                'p1': (20.0, y_pos),
                'p2': (70.0, y_pos),
                'p3': (140.0, y_pos),
                'p4': (190.0, y_pos)
            },
            'z_profile': {'z0': z0, 'z1': z1},
            'speed_profile': {'v0': 60.0, 'v1': 60.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    {color['name']:8s} @ Z={z0:4.1f}→{z1:4.1f} mm")
    
    # =========================================================================
    # Section 3: Crossing Strokes - Transparent Layering Demo
    # Shows CMY color mixing including same-color overlaps and order dependence
    # =========================================================================
    print("  Section 3: Transparent layering with crossings...")
    
    # Grid of horizontal strokes (all 7 colors)
    y_positions = [60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0]
    for i, (y_pos, color) in enumerate(zip(y_positions, colors)):
        stroke = {
            'id': f'horizontal-layer-{i}',
            'bezier': {
                'p1': (15.0, y_pos),
                'p2': (70.0, y_pos),
                'p3': (140.0, y_pos),
                'p4': (195.0, y_pos)
            },
            'z_profile': {'z0': 7.0, 'z1': 7.0},
            'speed_profile': {'v0': 60.0, 'v1': 60.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    Horizontal {color['name']:8s} @ Y={y_pos:5.1f} mm")
    
    # Grid of vertical strokes (all 7 colors) - will create same-color overlaps
    x_positions = [30.0, 55.0, 80.0, 105.0, 130.0, 155.0, 180.0]
    for i, (x_pos, color) in enumerate(zip(x_positions, colors)):
        stroke = {
            'id': f'vertical-layer-{i}',
            'bezier': {
                'p1': (x_pos, 30.0),
                'p2': (x_pos, 90.0),
                'p3': (x_pos, 190.0),
                'p4': (x_pos, 270.0)
            },
            'z_profile': {'z0': 7.0, 'z1': 7.0},
            'speed_profile': {'v0': 60.0, 'v1': 60.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    Vertical {color['name']:8s} @ X={x_pos:5.1f} mm")
    
    # =========================================================================
    # Section 3b: Order Dependence Test
    # Two pairs showing cyan-over-magenta vs magenta-over-cyan
    # =========================================================================
    print("  Section 3b: Order dependence test...")
    
    # Pair 1: Cyan first, then Magenta crosses it
    stroke = {
        'id': 'order-test-1a',
        'bezier': {'p1': (15.0, 35.0), 'p2': (40.0, 35.0), 'p3': (70.0, 35.0), 'p4': (95.0, 35.0)},
        'z_profile': {'z0': 9.0, 'z1': 9.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 1.0, 'm': 0.0, 'y': 0.0}  # cyan
    }
    canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
    
    stroke = {
        'id': 'order-test-1b',
        'bezier': {'p1': (55.0, 20.0), 'p2': (55.0, 30.0), 'p3': (55.0, 40.0), 'p4': (55.0, 50.0)},
        'z_profile': {'z0': 9.0, 'z1': 9.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.0, 'm': 1.0, 'y': 0.0}  # magenta
    }
    canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
    print("    Order test 1: cyan → magenta crosses")
    
    # Pair 2: Magenta first, then Cyan crosses it
    stroke = {
        'id': 'order-test-2a',
        'bezier': {'p1': (115.0, 35.0), 'p2': (140.0, 35.0), 'p3': (170.0, 35.0), 'p4': (195.0, 35.0)},
        'z_profile': {'z0': 9.0, 'z1': 9.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 0.0, 'm': 1.0, 'y': 0.0}  # magenta
    }
    canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
    
    stroke = {
        'id': 'order-test-2b',
        'bezier': {'p1': (155.0, 20.0), 'p2': (155.0, 30.0), 'p3': (155.0, 40.0), 'p4': (155.0, 50.0)},
        'z_profile': {'z0': 9.0, 'z1': 9.0},
        'speed_profile': {'v0': 50.0, 'v1': 50.0},
        'color_cmy': {'c': 1.0, 'm': 0.0, 'y': 0.0}  # cyan
    }
    canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
    print("    Order test 2: magenta → cyan crosses")
    
    # =========================================================================
    # Section 4: Curved Strokes - Bézier Demonstration
    # Different curve types with varying colors
    # =========================================================================
    print("  Section 4: Bézier curves...")
    
    curves = [
        # S-curve
        {'p1': (15.0, 50.0), 'p2': (50.0, 30.0), 'p3': (100.0, 70.0), 'p4': (135.0, 50.0)},
        # C-curve
        {'p1': (145.0, 40.0), 'p2': (170.0, 60.0), 'p3': (170.0, 100.0), 'p4': (145.0, 120.0)},
        # Wave
        {'p1': (15.0, 140.0), 'p2': (50.0, 120.0), 'p3': (100.0, 160.0), 'p4': (135.0, 140.0)},
    ]
    
    for i, curve in enumerate(curves):
        color = colors[(i + 5) % len(colors)]
        
        stroke = {
            'id': f'curve-{i}',
            'bezier': curve,
            'z_profile': {'z0': 7.0, 'z1': 7.0},
            'speed_profile': {'v0': 50.0, 'v1': 50.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    Curve {i+1} {color['name']:8s}")
    
    # =========================================================================
    # Save result
    # =========================================================================
    print("\nConverting to sRGB and saving...")
    output_path = Path("outputs/alcohol_ink_comprehensive_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert linear RGB to sRGB
    canvas_torch = torch.from_numpy(canvas).permute(2, 0, 1)
    canvas_srgb = color_utils.linear_to_srgb(canvas_torch)
    canvas_srgb = canvas_srgb.permute(1, 2, 0).numpy()
    
    # Convert to uint8
    canvas_uint8 = np.clip(canvas_srgb * 255, 0, 255).astype(np.uint8)
    
    # Save
    img = PIL.Image.fromarray(canvas_uint8, mode='RGB')
    img.save(output_path)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Canvas size: {W}x{H} pixels")
    # Total: 5 speed + 5 z-height + 7 horizontal + 7 vertical + 4 order-test + 3 curves = 31 strokes
    print(f"  Total strokes: {len(speed_pairs) + len(z_pairs) + 14 + 4 + len(curves)}")
    print(f"  Coverage: {np.mean(alpha)*100:.1f}%")
    print("\nDemo features:")
    print("  ✓ Within-stroke speed gradients (5 strokes: accelerating = dark→light)")
    print("  ✓ Within-stroke Z-height gradients (5 strokes: lifting = narrow→wide)")
    print("  ✓ Color variety (7 distinct CMY colors)")
    print("  ✓ Transparent layering (7×7 grid with same-color overlaps)")
    print("  ✓ Order dependence test (cyan→magenta vs magenta→cyan)")
    print("  ✓ Bézier curves (S-curve, C-curve, Wave)")
    print("  ✓ All strokes within A4 bounds (0-210mm x 0-297mm)")

if __name__ == '__main__':
    main()

