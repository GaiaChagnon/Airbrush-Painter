# Pen Tracer Quick Start Guide

**Purpose**: Extract black line art from images and generate pen tool paths for the Airbrush Painter system.

---

## Prerequisites

### Install Dependencies
```bash
pip install pyclipper shapely scikit-image
```

### Required Config Files
All configuration files are already set up in the repository:
- `configs/env_airbrush_v1.yaml` - Environment/canvas settings
- `configs/tools/pen_finetip_v1.yaml` - Pen tool parameters
- `configs/sim/pen_tracer_v2.yaml` - Edge detection & gamut-aware hatching settings

---

## Basic Usage

### Python API

```python
from src.data_pipeline import pen_tracer
from src.utils import validators

# Generate pen layer
result = pen_tracer.make_pen_layer(
    target_rgb_path="data/raw_images/my_image.jpg",
    env_cfg_path="configs/env_airbrush_v1.yaml",
    pen_tool_cfg_path="configs/tools/pen_finetip_v1.yaml",
    pen_tracer_cfg_path="configs/sim/pen_tracer_v2.yaml",
    out_dir="outputs/pen/my_job",
    cmy_canvas_path=None  # Optional: path to color layer for composite
)

# Access outputs
print(f"Pen vectors: {result['pen_vectors_yaml']}")
print(f"Preview: {result['pen_preview_png']}")
print(f"Composite: {result['composite_png']}")
print(f"Metrics: {result['metrics']}")
```

### Test Script

Use the provided test script:
```bash
python run_pen_tracer_test.py
```

This will process `data/raw_images/desktop-wallpaper-drawing-nature-blue-drawing.jpg` and save results to `outputs/pen_test/`.

---

## Output Files

### Directory Structure
```
outputs/pen/<job_name>/
├── pen_vectors.yaml          # Structured path data (YAML)
├── pen_preview.png           # Black strokes on white canvas
├── composite.png             # Pen layer over color layer (if provided)
└── debug/                    # Debug visualizations
    ├── 01_pen_mask_raw.png
    ├── 02_pen_mask_clean.png
    └── 03_distance_transform.png
```

### pen_vectors.yaml Schema
```yaml
schema: pen_vectors.v1
render_px: [width, height]
work_area_mm: [width_mm, height_mm]
paths:
  - id: "pen-000000"
    kind: "polyline"          # or "bezier"
    role: "outline"           # or "hatch", "fill"
    tip_diameter_mm: 0.3
    z_mm: -0.2
    feed_mm_s: 120.0
    points_mm:                # List of [x, y] coordinates in mm
      - [x1, y1]
      - [x2, y2]
      # ...
metadata:
  tool_name: "finetip_pigment_0.3mm"
  offset_mm: [12.50, -6.20, 0.00]
  generated_at: "2025-10-31T12:00:00Z"
  tracer_version: "pen_tracer.v2"
```

---

## Configuration Tuning

### Adjust Black Threshold

Edit `configs/sim/pen_tracer_v2.yaml`:
```yaml
thresholds:
  lab_l_max: 18.0    # Increase to capture more dark content (0-100)
  a_abs_min: 0.0     # Minimum chroma in a* axis
  b_abs_min: 0.0     # Minimum chroma in b* axis
```

**Guidelines**:
- `lab_l_max: 10-15` → Very dark blacks only
- `lab_l_max: 18-25` → Include dark grays
- `lab_l_max: 30+` → Capture mid-tones (may include shadows)

### Morphology Cleanup

```yaml
morphology:
  close_px: 1        # Fill small gaps (0-5)
  open_px: 0         # Remove small noise (0-3)
  min_area_px: 16    # Minimum component size (4-100)
```

### Path Generation

```yaml
filling:
  hatch_angles_deg: [0, 90, 45, 135]  # Hatching angles for regions
  hatch_spacing_scale: 1.00           # Spacing relative to tip diameter
  darkness_to_passes:                 # Overshading based on darkness
    - { l_max: 14.0, passes: 3 }
    - { l_max: 16.0, passes: 2 }
    - { l_max: 18.0, passes: 1 }
```

### Pen Tool Settings

Edit `configs/tools/pen_finetip_v1.yaml`:
```yaml
tip_diameter_mm: 0.30    # Pen tip width (0.1-2.0)
feed_mm_s: 120.0         # Drawing speed (10-300)
travel_mm_s: 250.0       # Travel speed (50-400)
draw_z_mm: -0.2          # Pen-down Z height (-2.0 to 10.0)
safe_z_mm: 5.0           # Safe Z for travel (0-30)
offset_mm: [12.50, -6.20, 0.00]  # Tool offset from airbrush
```

---

## Troubleshooting

### Problem: No black content extracted

**Symptoms**: `coverage_black: 0.0000` or very low coverage

**Solutions**:
1. Increase `lab_l_max` threshold (try 25-30)
2. Check debug images in `outputs/pen/<job>/debug/01_pen_mask_raw.png`
3. Verify image has actual black/dark content (not just blue/colored lines)

### Problem: Too much noise in output

**Symptoms**: Many small disconnected paths, high path count

**Solutions**:
1. Increase `min_area_px` (try 36-100)
2. Enable opening: `open_px: 1` or `2`
3. Reduce `lab_l_max` to capture only darkest content

### Problem: Gaps in line art

**Symptoms**: Warning "Max gap fraction > 0.02"

**Solutions**:
1. Increase `close_px` (try 2-3)
2. Reduce `simplify_tol_px` to preserve more detail
3. Check if original image has actual gaps (not a processing issue)

### Problem: Paths are too coarse

**Symptoms**: Jagged lines, lost detail

**Solutions**:
1. Reduce `simplify_tol_px` (try 0.5-1.0)
2. Check `render_px` resolution in environment config
3. Increase image resolution before processing

---

## Metrics Interpretation

| Metric | Typical Range | Meaning |
|--------|---------------|---------|
| `coverage_black` | 0.01-0.30 | Fraction of canvas covered by black strokes |
| `mean_luminance` | 0.70-0.99 | Average brightness (1.0 = white, 0.0 = black) |
| `max_gap_frac` | 0.00-0.10 | Largest gap as fraction of component size |
| `num_paths` | 10-1000 | Total number of generated paths |

**Notes**:
- Low `coverage_black` (<0.01) → Sparse line art or threshold too strict
- High `coverage_black` (>0.30) → Image is mostly dark/silhouette
- High `max_gap_frac` (>0.10) → Disconnected components or noise

---

## Advanced: Custom Pen Tool

Create a new pen tool config:

```yaml
# configs/tools/pen_custom.yaml
schema: pen_tool.v1
name: "custom_marker_0.5mm"
tip_diameter_mm: 0.50
overlap_frac: 0.30
max_passes: 3
feed_mm_s: 100.0
travel_mm_s: 200.0
safe_z_mm: 5.0
draw_z_mm: -0.3
plunge_mm_s: 10.0
offset_mm: [15.00, -8.00, 0.00]  # Measure from airbrush nozzle
macros:
  pen_up: "PEN_UP.gcode"
  pen_down: "PEN_DOWN.gcode"
accel:
  xy_mm_s2: 1500.0
joins:
  cap_style: "round"    # "round", "square", "flat"
  join_style: "round"   # "round", "miter", "bevel"
```

Then use it:
```python
result = pen_tracer.make_pen_layer(
    # ...
    pen_tool_cfg_path="configs/tools/pen_custom.yaml",
    # ...
)
```

---

## Integration with Paint Pipeline

The pen tracer is designed to run **after** CMY airbrush painting:

```python
# In scripts/paint.py (future integration)
from src.data_pipeline import pen_tracer
from src.utils import gcode_generator

# 1. Run CMY painting
cmy_result = run_cmy_painting(...)

# 2. Generate pen layer
pen_result = pen_tracer.make_pen_layer(
    target_rgb_path=target_image,
    env_cfg_path=env_cfg,
    pen_tool_cfg_path=pen_tool_cfg,
    pen_tracer_cfg_path=pen_tracer_cfg,
    out_dir=f"outputs/pen/{job_name}",
    cmy_canvas_path=cmy_result['final_canvas']  # Composite over color
)

# 3. Generate G-code
pen_vectors = validators.load_pen_vectors(pen_result['pen_vectors_yaml'])
pen_tool = validators.load_pen_tool_config(pen_tool_cfg)
machine = validators.load_machine_config(machine_cfg)

gcode_generator.generate_pen_gcode(
    pen_vectors,
    machine,
    pen_tool,
    output_path=f"gcode_output/{job_name}_pen.gcode"
)
```

---

## Performance Tips

1. **Large images**: Processing time scales linearly with pixel count
   - 1000×1000 px: ~1-2 seconds
   - 2000×2000 px: ~4-8 seconds
   - 4000×4000 px: ~15-30 seconds

2. **Memory**: Minimal (single image + masks in RAM)
   - 1000×1000 px: ~20 MB
   - 4000×4000 px: ~300 MB

3. **Optimization**:
   - Disable debug output: `save_intermediates: false`
   - Reduce morphology iterations
   - Increase `min_area_px` to filter small components early

---

## References

- **Schema Definitions**: `configs/schema/pen_*.v1.yaml`
- **Implementation**: `src/data_pipeline/pen_tracer.py`
- **Validators**: `src/utils/validators.py` (Pydantic models)
- **Test Success Report**: `documentation/PEN_TRACER_TEST_SUCCESS.md`
- **Full Design Doc**: `documentation/PEN_TRACER_IMPLEMENTATION.md`

---

**Last Updated**: 2025-10-31  
**Version**: pen_tracer.v2

