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
    
    # Define color palette (7 distinct colors in CMY space)
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
    # Section 1: Speed Variation (horizontal strokes, top section)
    # Fast → Slow: lighter → darker
    # =========================================================================
    print("  Section 1: Speed variation...")
    speeds = [180.0, 120.0, 80.0, 50.0, 25.0]
    y_start = 30.0
    y_spacing = 25.0
    
    for i, speed in enumerate(speeds):
        color = colors[i % len(colors)]
        y_pos = y_start + i * y_spacing
        
        stroke = {
            'id': f'speed-{speed}',
            'bezier': {
                'p1': (20.0, y_pos),
                'p2': (70.0, y_pos),
                'p3': (140.0, y_pos),
                'p4': (190.0, y_pos)
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': speed, 'v1': speed},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    {color['name']:8s} @ {speed:5.1f} mm/s")
    
    # =========================================================================
    # Section 2: Z-Height Variation (horizontal strokes, middle section)
    # Low Z → High Z: narrow → wide
    # =========================================================================
    print("  Section 2: Z-height variation...")
    z_heights = [3.0, 5.0, 8.0, 12.0, 16.0]
    y_start = 160.0
    y_spacing = 25.0
    
    for i, z_height in enumerate(z_heights):
        color = colors[(i + 2) % len(colors)]
        y_pos = y_start + i * y_spacing
        
        stroke = {
            'id': f'z-{z_height}',
            'bezier': {
                'p1': (20.0, y_pos),
                'p2': (70.0, y_pos),
                'p3': (140.0, y_pos),
                'p4': (190.0, y_pos)
            },
            'z_profile': {'z0': z_height, 'z1': z_height},
            'speed_profile': {'v0': 60.0, 'v1': 60.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    {color['name']:8s} @ Z={z_height:4.1f} mm")
    
    # =========================================================================
    # Section 3: Crossing Strokes - Transparent Layering Demo
    # Vertical and diagonal strokes that intersect to show blending
    # =========================================================================
    print("  Section 3: Transparent layering with crossings...")
    
    # Vertical strokes (left to right)
    x_positions = [35.0, 65.0, 95.0, 125.0, 155.0, 185.0]
    for i, x_pos in enumerate(x_positions):
        color = colors[i % len(colors)]
        
        stroke = {
            'id': f'vertical-{i}',
            'bezier': {
                'p1': (x_pos, 25.0),
                'p2': (x_pos, 80.0),
                'p3': (x_pos, 200.0),
                'p4': (x_pos, 270.0)
            },
            'z_profile': {'z0': 8.0, 'z1': 8.0},
            'speed_profile': {'v0': 70.0, 'v1': 70.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    Vertical {color['name']:8s} @ X={x_pos:5.1f} mm")
    
    # Diagonal strokes (crossing pattern)
    diagonals = [
        # Top-left to bottom-right
        {'p1': (15.0, 25.0), 'p2': (70.0, 100.0), 'p3': (140.0, 200.0), 'p4': (195.0, 270.0)},
        # Bottom-left to top-right
        {'p1': (15.0, 270.0), 'p2': (70.0, 200.0), 'p3': (140.0, 100.0), 'p4': (195.0, 25.0)},
    ]
    
    for i, diag in enumerate(diagonals):
        color = colors[(i + 4) % len(colors)]
        
        stroke = {
            'id': f'diagonal-{i}',
            'bezier': diag,
            'z_profile': {'z0': 10.0, 'z1': 10.0},
            'speed_profile': {'v0': 80.0, 'v1': 80.0},
            'color_cmy': {k: v for k, v in color.items() if k in ['c', 'm', 'y']}
        }
        canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
        print(f"    Diagonal {color['name']:8s}")
    
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
    print(f"  Total strokes: {len(speeds) + len(z_heights) + len(x_positions) + len(diagonals) + len(curves)}")
    print(f"  Coverage: {np.mean(alpha)*100:.1f}%")
    print("\nDemo features:")
    print("  ✓ Speed variation (5 strokes: fast→slow = light→dark)")
    print("  ✓ Z-height variation (5 strokes: low→high = narrow→wide)")
    print("  ✓ Color variety (7 distinct CMY colors)")
    print("  ✓ Transparent layering (vertical + diagonal crossings)")
    print("  ✓ Bézier curves (S-curve, C-curve, Wave)")
    print("  ✓ All strokes within A4 bounds (0-210mm x 0-297mm)")

if __name__ == '__main__':
    main()

