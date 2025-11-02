#!/usr/bin/env python3
"""Test pen tracer - gamut-aware A4 print quality."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import pen_tracer
from src.utils import logging_config

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Test pen tracer on images")
parser.add_argument("--input", "-i", type=str, 
                    default="data/raw_images/desktop-wallpaper-drawing-nature-blue-drawing.jpg",
                    help="Input image path or directory")
parser.add_argument("--output", "-o", type=str, 
                    default="outputs/pen_test",
                    help="Output directory")
parser.add_argument("--env-cfg", type=str, 
                    default="configs/env_airbrush_v1.yaml",
                    help="Environment config path")
parser.add_argument("--pen-tool-cfg", type=str, 
                    default="configs/tools/pen_finetip_v1.yaml",
                    help="Pen tool config path")
parser.add_argument("--pen-tracer-cfg", type=str, 
                    default="configs/sim/pen_tracer_v2.yaml",
                    help="Pen tracer config path")
parser.add_argument("--cmy-canvas", type=str, default=None,
                    help="Optional CMY canvas image path")

args = parser.parse_args()

# Setup logging
Path("outputs/logs").mkdir(parents=True, exist_ok=True)
logging_config.setup_logging(log_level="INFO", log_file="outputs/logs/pen_test.log")

# Determine input paths
input_path = Path(args.input)
if input_path.is_dir():
    # Process all images in directory
    image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    image_paths = [p for p in image_paths if not p.name.startswith('.')]
else:
    # Single image
    image_paths = [input_path]

env_cfg_path = args.env_cfg
pen_tool_cfg_path = args.pen_tool_cfg
pen_tracer_cfg_path = args.pen_tracer_cfg
cmy_canvas_path = args.cmy_canvas

print("="*80)
print("PEN TRACER TEST - A4 Print Quality")
print("="*80)
print(f"\nProcessing {len(image_paths)} image(s)")
print(f"Output directory: {args.output}")
print("\nSpecifications:")
print("  • Paper size: A4 (210mm × 297mm)")
print("  • Resolution: 300 DPI (2480 × 3508 pixels)")
print("  • Pen tip: 0.3mm fine liner")
print("  • Max coverage: 20% (sparse hatching)")
print("  • Gamut-aware: Only out-of-CMY colors")
print("  • Single-direction hatching: 45° diagonal")
print("="*80)

results = []
errors = []

for idx, image_path in enumerate(image_paths, 1):
    print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path.name}")
    print("-" * 80)
    
    # Create output directory for this image
    image_stem = image_path.stem
    out_dir = Path(args.output) / image_stem
    
    try:
        result = pen_tracer.make_pen_layer(
            target_rgb_path=str(image_path),
            env_cfg_path=env_cfg_path,
            pen_tool_cfg_path=pen_tool_cfg_path,
            pen_tracer_cfg_path=pen_tracer_cfg_path,
            out_dir=str(out_dir),
            cmy_canvas_path=cmy_canvas_path
        )
        
        results.append({
            'image': image_path.name,
            'result': result
        })
        
        print(f"✓ SUCCESS: {image_path.name}")
        print(f"  • Edges: {result['metrics']['num_edge_paths']}")
        print(f"  • Hatches: {result['metrics']['num_hatch_paths']}")
        print(f"  • Coverage: {result['metrics']['coverage_black']*100:.1f}%")
        print(f"  • Output: {out_dir}")
        
    except Exception as e:
        errors.append({
            'image': image_path.name,
            'error': str(e)
        })
        print(f"✗ ERROR: {image_path.name}")
        print(f"  {type(e).__name__}: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nProcessed: {len(results)}/{len(image_paths)} images")
if errors:
    print(f"Errors: {len(errors)}")
    for err in errors:
        print(f"  • {err['image']}: {err['error']}")

if results:
    print("\nSuccessful outputs:")
    for r in results:
        metrics = r['result']['metrics']
        print(f"\n{r['image']}:")
        print(f"  • Resolution: {metrics['resolution'][0]}×{metrics['resolution'][1]} px")
        print(f"  • Edges: {metrics['num_edge_paths']}, Hatches: {metrics['num_hatch_paths']}")
        print(f"  • Coverage: {metrics['coverage_black']*100:.1f}%")
        print(f"  • Preview: {r['result']['pen_preview_png']}")
        print(f"  • Composite: {r['result']['composite_png']}")

print("\n" + "="*80)
print("All results saved to:", args.output)
print("="*80)

sys.exit(0 if not errors else 1)

