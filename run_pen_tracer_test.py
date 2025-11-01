#!/usr/bin/env python3
"""Test pen tracer - gamut-aware A4 print quality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import pen_tracer
from src.utils import logging_config

# Setup logging
Path("outputs/logs").mkdir(parents=True, exist_ok=True)
logging_config.setup_logging(log_level="INFO", log_file="outputs/logs/pen_test.log")

# Paths
image_path = "data/raw_images/desktop-wallpaper-drawing-nature-blue-drawing.jpg"
env_cfg_path = "configs/env_airbrush_v1.yaml"
pen_tool_cfg_path = "configs/tools/pen_finetip_v1.yaml"
pen_tracer_cfg_path = "configs/sim/pen_tracer_v2.yaml"
out_dir = "outputs/pen_test_final"

print("="*80)
print("PEN TRACER TEST - A4 Print Quality")
print("="*80)
print(f"\nInput: {image_path}")
print(f"Output: {out_dir}")
print("\nSpecifications:")
print("  • Paper size: A4 (210mm × 297mm)")
print("  • Resolution: 300 DPI (2480 × 3508 pixels)")
print("  • Pen tip: 0.3mm fine liner")
print("  • Max coverage: 20% (sparse hatching)")
print("  • Gamut-aware: Only out-of-CMY colors")
print("  • Single-direction hatching: 45° diagonal")
print("="*80)

try:
    result = pen_tracer.make_pen_layer_v3(
        target_rgb_path=image_path,
        env_cfg_path=env_cfg_path,
        pen_tool_cfg_path=pen_tool_cfg_path,
        pen_tracer_cfg_path=pen_tracer_cfg_path,
        out_dir=out_dir,
        cmy_canvas_path=None
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • Pen vectors: {result['pen_vectors_yaml']}")
    print(f"  • Pen preview: {result['pen_preview_png']}")
    print(f"  • Composite: {result['composite_png']}")
    
    print("\nMetrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.4f}")
        else:
            print(f"  • {key}: {value}")
    
    print("\n" + "="*80)
    print("Print Quality Stats:")
    print(f"  • Resolution: {result['metrics']['resolution'][0]}×{result['metrics']['resolution'][1]} pixels")
    print(f"  • DPI: ~{result['metrics']['resolution'][1]/297*25.4:.0f} (height-based)")
    print(f"  • Total paths: {result['metrics']['num_paths']}")
    print(f"  • Edge outlines: {result['metrics']['num_edge_paths']}")
    print(f"  • Hatch lines: {result['metrics']['num_hatch_paths']}")
    print(f"  • Coverage: {result['metrics']['coverage_black']*100:.1f}%")
    print("="*80)
    print(f"\nView results: {out_dir}/pen_preview.png")
    print("\nReady for A4 printing or G-code generation!")
    
except Exception as e:
    print("\n" + "="*80)
    print("ERROR!")
    print("="*80)
    print(f"\n{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

