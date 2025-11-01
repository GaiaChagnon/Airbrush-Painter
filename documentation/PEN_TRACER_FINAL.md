# Pen Tracer - Final Production Implementation

**Date**: 2025-11-01  
**Status**: ✅ **Production Ready**

---

## Overview

The pen tracer is a gamut-aware black pen layer generator that complements CMY airbrush painting. It extracts edge outlines and fills dark regions that CMY cannot reproduce, producing sparse single-direction hatching that leaves room for color.

---

## File Structure

### Production Code
```
src/data_pipeline/
└── pen_tracer.py              # Main implementation (formerly pen_tracer_v3.py)

configs/
├── schema/
│   ├── pen_tool.v1.yaml       # Pen tool schema definition
│   ├── pen_tracer.v1.yaml     # Old tracer schema (deprecated)
│   ├── pen_tracer.v2.yaml     # Current tracer schema
│   └── pen_vectors.v1.yaml    # Output format schema
├── tools/
│   └── pen_finetip_v1.yaml    # 0.3mm fine liner config
└── sim/
    └── pen_tracer_v2.yaml     # Production tracer config

src/utils/
└── validators.py              # Pydantic models for all schemas

gcode/macros/
├── PEN_UP.gcode               # Pen lift macro
└── PEN_DOWN.gcode             # Pen lower macro
```

### Test & Demo
```
run_pen_tracer_test.py         # A4 print quality test script
data/raw_images/
└── desktop-wallpaper-drawing-nature-blue-drawing.jpg  # Test image

outputs/
└── pen_test_final/            # Latest test output
    ├── pen_vectors.yaml       # 3,104 paths
    ├── pen_preview.png        # 2184×3508 A4 preview
    ├── composite.png          # Final composite
    └── debug/                 # Intermediate masks
```

### Documentation
```
documentation/
├── PEN_TRACER_FINAL.md        # This file
├── PEN_TRACER_IMPLEMENTATION.md  # Original design doc
├── PEN_TRACER_V2_SUCCESS.md   # V2 development notes
└── PEN_TRACER_QUICKSTART.md   # User guide
```

---

## Removed Files (Cleanup)

### Obsolete Implementations
- ❌ `src/data_pipeline/pen_tracer.py` (V1 - black-only thresholding)
- ❌ `src/data_pipeline/pen_tracer_v2.py` (V2 - over-hatching)
- ✅ Renamed `pen_tracer_v3.py` → `pen_tracer.py` (production)

### Test Scripts
- ❌ `run_pen_tracer_test.py` (V1 test)
- ❌ `run_pen_tracer_v2_test.py` (V2 test)
- ❌ `run_pen_tracer_v2_quick.py` (V2 quick test)
- ❌ `create_comparison.py` (development comparison)
- ✅ Renamed `run_pen_tracer_v3_test.py` → `run_pen_tracer_test.py`

### Test Outputs
- ❌ `outputs/pen_test/` (V1 output)
- ❌ `outputs/pen_test_v2/` (V2 output)
- ❌ `outputs/pen_test_v2_quick/` (V2 quick output)
- ✅ Renamed `outputs/pen_test_v3/` → `outputs/pen_test_final/`

---

## Production Specifications

### A4 Print Quality
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Paper Size** | 210×297mm | A4 portrait |
| **Resolution** | 2184×3508 pixels | ~300 DPI |
| **Pen Width** | 0.3mm | Fine liner |
| **Coverage** | 9.8% (max 20%) | Sparse hatching |
| **Total Paths** | 3,104 | 1,380 edges + 1,724 hatches |

### Configuration (`pen_tracer_v2.yaml`)
```yaml
schema: pen_tracer.v2

output:
  target_height_px: 3508  # A4 @ 300 DPI

edge_detection:
  enabled: true
  canny_low: 50.0
  canny_high: 150.0
  min_length_px: 20
  simplify_tol_px: 1.5

shadow_hatching:
  enabled: true
  gamut_aware: true
  
  cmy_gamut:
    min_luminance: 15.0    # Darkest CMY can achieve
    max_chroma: 80.0       # Max saturation
  
  darkness_levels:
    - l_min: 0.0           # Exclusive ranges
      l_max: 30.0          # No overlap!
      passes: 1
      hatch_angles: [45]   # Single direction
    
    - l_min: 30.0
      l_max: 50.0
      passes: 1
      hatch_angles: [45]
    
    - l_min: 50.0
      l_max: 70.0
      passes: 1
      hatch_angles: [45]
  
  min_area_px: 500
  spacing_scale: 2.5       # Wide spacing
  max_hatch_coverage: 0.20 # 20% limit

calibration:
  calibration_file: null   # Ready for real gamut data
  margin: 0.05
```

---

## Key Features

### ✅ Gamut-Aware Hatching
- Only hatches regions outside CMY gamut (17.88% of image)
- Uses LAB color space for accurate gamut calculation
- Ready to load real calibration data from airbrush tests

### ✅ No Double Tracing
- Exclusive darkness ranges (L* 0-30, 30-50, 50-70)
- Each pixel hatched only once
- Prevents overlapping/over-inking

### ✅ Coverage Limits
- Configurable maximum coverage (default 20%)
- Tracks actual coverage during generation
- Stops when limit reached

### ✅ Single-Direction Hatching
- 45° diagonal lines only
- Leaves space for CMY colors to show through
- No cross-hatching

### ✅ A4 Print Quality
- 300 DPI resolution
- Maintains aspect ratio
- Proper mm/px scaling for G-code

---

## Usage

### Basic Test
```bash
python run_pen_tracer_test.py
```

### Python API
```python
from src.data_pipeline import pen_tracer

result = pen_tracer.make_pen_layer_v3(
    target_rgb_path="image.jpg",
    env_cfg_path="configs/env_airbrush_v1.yaml",
    pen_tool_cfg_path="configs/tools/pen_finetip_v1.yaml",
    pen_tracer_cfg_path="configs/sim/pen_tracer_v2.yaml",
    out_dir="outputs/pen/my_job"
)

# Outputs
print(result['pen_vectors_yaml'])  # YAML with all paths
print(result['pen_preview_png'])   # Preview image
print(result['composite_png'])     # Final composite
print(result['metrics'])           # Coverage, path counts
```

### Integration with Paint Pipeline
```python
# After CMY painting
pen_result = pen_tracer.make_pen_layer_v3(
    target_rgb_path=target_image,
    env_cfg_path=env_cfg,
    pen_tool_cfg_path=pen_tool_cfg,
    pen_tracer_cfg_path=pen_tracer_cfg,
    out_dir=f"outputs/pen/{job_name}",
    cmy_canvas_path=cmy_final_canvas  # Composite over color
)

# Generate G-code
pen_vectors = validators.load_pen_vectors(pen_result['pen_vectors_yaml'])
pen_tool = validators.load_pen_tool_config(pen_tool_cfg)
machine = validators.load_machine_config(machine_cfg)

gcode_generator.generate_pen_gcode(
    pen_vectors, machine, pen_tool,
    output_path=f"gcode_output/{job_name}_pen.gcode"
)
```

---

## Tuning Guide

### Adjust Coverage
```yaml
max_hatch_coverage: 0.15  # Reduce to 15% for sparser hatching
```

### Adjust Darkness Threshold
```yaml
cmy_gamut:
  min_luminance: 20.0  # Increase if CMY can't get dark enough
```

### Adjust Hatch Spacing
```yaml
spacing_scale: 3.0  # Increase for wider spacing
```

### Change Resolution
```yaml
output:
  target_height_px: 4096  # 4K quality
  # or
  target_height_px: null  # Use original resolution
```

---

## Metrics

### Test Image Results
- **Input**: 850×1365 landscape painting
- **Output**: 2184×3508 A4 @ 300 DPI
- **Out-of-gamut**: 17.88% (only these regions hatched)
- **Actual coverage**: 9.8% (under 20% limit)
- **Edge contours**: 1,380
- **Hatch lines**: 1,724
- **Total paths**: 3,104
- **Processing time**: ~60 seconds

---

## Next Steps

### Immediate
- [x] Clean up old implementations
- [x] Rename V3 to production
- [x] Update documentation
- [ ] Add to main `paint.py` pipeline
- [ ] Generate G-code for test image

### Future Enhancements
- [ ] Load real CMY gamut from calibration
- [ ] Add path ordering (TSP optimization)
- [ ] Support multiple hatch angles per level
- [ ] Add stippling mode for mid-tones
- [ ] Adaptive hatching (follow contours)

---

## Conclusion

The pen tracer is **production-ready** for A4 print quality output. It intelligently complements CMY airbrush by:
- Only hatching what CMY cannot reproduce (gamut-aware)
- Using sparse single-direction hatching (leaves room for color)
- Enforcing coverage limits (prevents over-inking)
- Producing clean edge outlines (captures all detail)

**Ready for integration into the main painting pipeline and G-code generation.**

