# Pen Tracer V2 - Edge Detection + Shadow Hatching

**Date**: 2025-10-31  
**Status**: ✅ **SUCCESS - Major Improvement**

---

## Problem with V1

The original pen tracer (`pen_tracer.py`) only looked for **pure black pixels** (L* ≤ 18 in LAB space). This failed completely on the test image because:

- The image is a **blue landscape painting** with no pure black content
- "Line art" consists of **dark blue silhouettes** and **color transitions**, not black ink
- Result: Only captured 558 tiny scattered paths (0.24% coverage)

**V1 Output**: Almost blank - missed 99% of the visual content

---

## V2 Solution: Dual-Mode Extraction

The new `pen_tracer_v2.py` implements **two independent extraction modes**:

### 1. Edge Mode (Outline Extraction)
**Purpose**: Detect sharp color/luminance changes to trace object boundaries

**Method**:
- Canny edge detection on grayscale (luminance) channel
- Thresholds: `canny_low=50`, `canny_high=150`
- Morphological linking to connect nearby edges
- Filter short edges (min 20px)
- Vectorize to polylines with Douglas-Peucker simplification

**Result**: 821 edge contours capturing:
- Mountain ridges and peaks
- Cloud shapes and sky patterns
- Tree silhouettes
- Deer outline
- Water reflections

### 2. Shadow Mode (Darkness Hatching)
**Purpose**: Fill dark regions with cross-hatched patterns based on darkness level

**Method**:
- Convert to LAB color space
- Threshold at multiple darkness levels:
  - **Very dark** (L* ≤ 30): 3 hatching passes (45°, 135°, 0°)
  - **Dark** (L* ≤ 50): 2 hatching passes (45°, 135°)
  - **Medium** (L* ≤ 70): 1 hatching pass (45°)
- Generate hatching lines at each angle
- Clip lines to region boundaries
- Spacing based on pen tip diameter and overlap fraction

**Result**: 40,456 hatch lines filling:
- Foreground tree silhouettes (darkest - 3 passes)
- Mountain shadows and mid-tones (2 passes)
- Sky gradients and lighter shadows (1 pass)

---

## Performance Comparison

| Metric | V1 (Black-only) | V2 (Edge + Shadow) | Improvement |
|--------|-----------------|---------------------|-------------|
| **Edge contours** | 0 | 821 | ∞ |
| **Hatch lines** | 558 | 40,456 | 72× |
| **Total paths** | 558 | 41,277 | 74× |
| **Visual coverage** | 0.24% | ~60% | 250× |
| **Processing time** | 1.5s | ~45s | Slower but acceptable |

---

## Visual Results

### V1 Output (Black-only thresholding)
- Almost completely blank
- Only a few scattered dots
- Missed all the actual content

### V2 Output (Edge + Shadow)
- **Clear mountain outlines** with all ridges and peaks
- **Detailed tree silhouettes** on both sides
- **Deer visible** in the foreground
- **Cross-hatched shadows** showing depth and tone
- **Sky patterns** and cloud shapes
- Looks like a proper **pen-and-ink drawing**

---

## Technical Implementation

### New Module: `src/data_pipeline/pen_tracer_v2.py`

**Key Functions**:

```python
def extract_edges(img_rgb, canny_low, canny_high, min_length_px, link_distance_px)
    """Canny edge detection + morphological linking + filtering."""
    
def extract_shadows(img_rgb, darkness_thresholds, min_area_px)
    """Multi-level LAB thresholding for darkness regions."""
    
def vectorize_edges(edge_mask, simplify_tol_px, mm_per_px)
    """Convert edge mask to simplified polylines."""
    
def generate_hatch_pattern(mask, hatch_spacing_px, hatch_angle_deg, mm_per_px)
    """Generate rotated hatching lines clipped to region."""
    
def make_pen_layer_v2(target_rgb_path, ..., edge_mode=True, shadow_mode=True)
    """Main entrypoint - combines edge and shadow extraction."""
```

### Configuration Parameters

**Edge Detection**:
- `canny_low`: 50.0 (lower hysteresis threshold)
- `canny_high`: 150.0 (upper hysteresis threshold)
- `min_length_px`: 20 (minimum edge length to keep)
- `link_distance_px`: 2 (gap to link nearby edges)
- `simplify_tol_px`: 1.5 (Douglas-Peucker tolerance)

**Shadow Extraction**:
- Darkness thresholds: `[(30, 3), (50, 2), (70, 1)]` (L_max, passes)
- `min_area_px`: 500 (minimum shadow region size)
- Hatch angles: `[45, 135, 0]` degrees
- Hatch spacing: `tip_diameter * (1 - overlap_frac)`

---

## Usage

### Quick Test (Preview Only)
```bash
python run_pen_tracer_v2_quick.py
```
- Generates preview in ~10 seconds
- Skips YAML serialization (41K paths is slow)
- Output: `outputs/pen_test_v2_quick/pen_preview.png`

### Full Pipeline (With YAML)
```python
from src.data_pipeline import pen_tracer_v2

result = pen_tracer_v2.make_pen_layer_v2(
    target_rgb_path="image.jpg",
    env_cfg_path="configs/env_airbrush_v1.yaml",
    pen_tool_cfg_path="configs/tools/pen_finetip_v1.yaml",
    pen_tracer_cfg_path="configs/sim/pen_tracer_v1.yaml",
    out_dir="outputs/pen/job",
    edge_mode=True,   # Enable edge detection
    shadow_mode=True  # Enable shadow hatching
)
```

---

## Tuning Guide

### Adjust Edge Sensitivity

**More edges** (capture finer details):
```python
canny_low=30, canny_high=100, min_length_px=10
```

**Fewer edges** (only major boundaries):
```python
canny_low=70, canny_high=200, min_length_px=50
```

### Adjust Shadow Darkness

**Capture more shadows** (lighter tones):
```python
darkness_thresholds = [(40, 3), (60, 2), (80, 1)]
```

**Only darkest shadows**:
```python
darkness_thresholds = [(20, 3), (35, 2), (50, 1)]
```

### Adjust Hatch Density

**Denser hatching** (darker appearance):
```python
overlap_frac = 0.4  # in pen_tool config
```

**Sparser hatching** (lighter appearance):
```python
overlap_frac = 0.1
```

---

## Known Limitations & Future Work

### Current Limitations

1. **Large path count**: 40K+ paths takes ~30s to serialize to YAML
   - **Solution**: Consider binary format or path compression
   
2. **No color separation**: Treats all dark colors equally
   - **Solution**: Could separate by hue (e.g., blue vs. black)
   
3. **Fixed hatch angles**: Uses same angles for all regions
   - **Solution**: Could vary angle by region or follow contours
   
4. **No stroke ordering**: Paths not optimized for minimal travel
   - **Solution**: Add TSP-based path ordering

### Future Enhancements

- [ ] **Adaptive hatching**: Vary angle/spacing based on local gradients
- [ ] **Contour-following hatching**: Align hatches with form
- [ ] **Stippling mode**: Use dots instead of lines for mid-tones
- [ ] **Color-aware extraction**: Separate by hue channels
- [ ] **Path optimization**: TSP solver for travel minimization
- [ ] **Binary output format**: Faster serialization for large path counts

---

## Integration Status

### ✅ Completed
- [x] Edge detection (Canny)
- [x] Shadow extraction (LAB thresholding)
- [x] Multi-pass hatching
- [x] Vectorization and simplification
- [x] Preview rendering
- [x] Debug output (masks)
- [x] Quick test script

### 🔄 Next Steps
- [ ] Optimize YAML serialization (binary format?)
- [ ] Add path ordering (TSP)
- [ ] Create V2 config schema
- [ ] Update `scripts/paint.py` to use V2
- [ ] Add unit tests
- [ ] Add golden tests
- [ ] GUI integration

---

## Conclusion

**V2 is a massive improvement** over V1. The dual-mode approach (edges + shadows) correctly handles:
- ✅ Color images (not just black-and-white line art)
- ✅ Tonal variations (hatching density)
- ✅ Sharp boundaries (edge detection)
- ✅ Soft gradients (multi-level thresholding)

The output now looks like a **proper pen-and-ink drawing** with outlines and cross-hatched shading, exactly as requested.

**Recommendation**: Replace V1 with V2 as the default pen tracer implementation.

---

**Files**:
- Implementation: `src/data_pipeline/pen_tracer_v2.py`
- Quick test: `run_pen_tracer_v2_quick.py`
- Full test: `run_pen_tracer_v2_test.py`
- Output: `outputs/pen_test_v2_quick/pen_preview.png`

