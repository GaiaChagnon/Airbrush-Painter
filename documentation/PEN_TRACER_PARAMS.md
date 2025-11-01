# Pen Tracer Parameter Guide

**Version:** V2 (Gamut-Aware)  
**Last Updated:** 2025-11-01  
**Config Schema:** `pen_tracer.v2`

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Output Resolution](#output-resolution)
3. [Edge Detection](#edge-detection)
4. [Shadow Hatching](#shadow-hatching)
5. [Calibration](#calibration)
6. [Common Tuning Scenarios](#common-tuning-scenarios)
7. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Default Configuration (A4 Print Quality, Sparse Hatching)
```yaml
output:
  target_height_px: 3508  # A4 @ 300 DPI
edge_detection:
  canny_low: 100.0        # Hard edges only
  canny_high: 200.0
shadow_hatching:
  spacing_scale: 2.5      # Sparse hatching
  max_hatch_coverage: 0.20  # 20% max coverage
```

### Key Performance Knobs
| To achieve...           | Adjust...                | Direction  |
|-------------------------|--------------------------|------------|
| **Faster processing**   | `target_height_px`       | Lower      |
| **Fewer paths**         | `canny_low`/`canny_high` | Higher     |
| **Lighter hatching**    | `spacing_scale`          | Higher     |
| **Sparser output**      | `max_hatch_coverage`     | Lower      |
| **Cleaner outlines**    | `min_length_px`          | Higher     |

---

## Output Resolution

### `output.target_height_px`
**Controls:** Final output height in pixels. Width is computed to maintain aspect ratio.

#### Common Presets

| Value  | DPI | Paper Size | Quality Level       | Typical Use                |
|--------|-----|------------|---------------------|----------------------------|
| `3508` | 300 | A4 height  | Professional        | Final prints, exhibitions  |
| `2339` | 200 | A4 height  | Good                | Proofs, test prints        |
| `1754` | 150 | A4 height  | Draft               | Quick previews, iteration  |
| `1170` | 100 | A4 height  | Low                 | Speed tests, development   |
| `null` | -   | env config | Simulation override | Testing with env configs   |

#### Performance Impact
- **3508px:** ~2-5 seconds processing, 2,000-5,000 paths
- **2339px:** ~1-2 seconds processing, 1,000-3,000 paths
- **1754px:** ~0.5-1 second processing, 500-1,500 paths

#### Disk Space (approximate)
- **YAML vectors:** 50-200 KB per 1,000 paths
- **Preview PNG:** ~5-15 MB at 3508px, ~2-5 MB at 2339px
- **G-code:** ~100-500 KB per 1,000 paths

---

### `output.min_px` / `output.max_px`
**Controls:** Safety bounds (clamping limits).

- **`min_px: 512`**: Prevents invalid/tiny outputs
- **`max_px: 4096`**: Prevents out-of-memory errors

**Recommendation:** Leave at defaults unless you have specific hardware constraints.

---

## Edge Detection

Edge detection extracts **outlines** from sharp color transitions in the image.

### Core Parameters

#### `edge_detection.canny_low` / `edge_detection.canny_high`
**Controls:** Canny edge detector sensitivity (applied to luminance channel).

**Algorithm:** Canny hysteresis thresholding
- Pixels with gradient magnitude > `canny_high` are **strong edges**
- Pixels with gradient magnitude > `canny_low` (and connected to strong edges) are **weak edges**
- Typical ratio: `canny_high = 2-3× canny_low`

#### Tuning Guide

| Style Goal                  | `canny_low` | `canny_high` | Effect                                     |
|-----------------------------|-------------|--------------|--------------------------------------------|
| **Clean, essential outlines** | 100-150     | 200-300      | Only hard boundaries (dark vs light)       |
| **Balanced detail**         | 70-100      | 150-200      | Moderate detail, some gradients (default)  |
| **Fine detail, texture**    | 40-70       | 100-150      | Captures soft gradients, busy output       |
| **Watercolor-style**        | 30-50       | 80-120       | All color changes, very detailed           |

#### Visual Examples

```yaml
# Scenario 1: Minimalist illustration (bold outlines only)
canny_low: 120.0
canny_high: 250.0
# Result: ~500-1,000 edge paths, clean silhouettes

# Scenario 2: Balanced (current default)
canny_low: 100.0
canny_high: 200.0
# Result: ~1,000-2,000 edge paths, good balance

# Scenario 3: Detailed technical drawing
canny_low: 50.0
canny_high: 150.0
# Result: ~2,000-5,000 edge paths, captures texture
```

---

### `edge_detection.min_length_px`
**Controls:** Minimum edge contour length (in pixels). Shorter edges are discarded.

**Purpose:** Remove noise, texture, and insignificant details.

| Value    | Effect                                   | Typical Use              |
|----------|------------------------------------------|--------------------------|
| `5-10`   | Keep fine details, texture               | High-detail illustrations |
| `15-25`  | Balanced (current: `20`)                 | General use              |
| `30-50`  | Only major features, coarse outlines     | Minimalist/abstract art  |

**Trade-off:**
- Higher → fewer, cleaner paths, but may miss important small details
- Lower → more detailed, but noisier output

---

### `edge_detection.link_distance_px`
**Controls:** Maximum gap distance for connecting broken edge contours.

**Purpose:** Bridge small gaps in edges (from image compression, noise, etc.).

| Value  | Effect                                      |
|--------|---------------------------------------------|
| `0`    | No linking (fragmented edges)               |
| `2-5`  | Moderate linking (current: `2`)             |
| `8+`   | Aggressive linking (may join unrelated edges) |

**Recommendation:** 2-5 is typically sufficient. Increase if you see many broken outlines.

---

### `edge_detection.simplify_tol_px`
**Controls:** Douglas-Peucker simplification tolerance (in pixels).

**Purpose:** Reduce number of points per path while maintaining shape fidelity.

| Value      | Effect                                     | G-code Size      |
|------------|--------------------------------------------|------------------|
| `0.5-1.0`  | High fidelity, many points                 | Large (~500 KB)  |
| `1.5-2.5`  | Balanced (current: `1.5`)                  | Medium (~200 KB) |
| `3.0-5.0`  | Aggressive smoothing, angular appearance   | Small (~100 KB)  |

**Trade-off:**
- Higher → smaller files, smoother curves, faster G-code execution
- Lower → more faithful to original, larger files

---

## Shadow Hatching

Shadow hatching fills dark regions that the CMY airbrush cannot reproduce accurately.

### Philosophy: Gamut-Aware Hatching
**Key Concept:** Only hatch regions **outside** the CMY gamut.

- **Inside gamut:** CMY can handle it → no pen needed
- **Outside gamut:** CMY cannot reproduce → pen hatching required

This prevents double-tracing and over-inking while preserving color vibrancy.

---

### CMY Gamut Definition

#### `shadow_hatching.cmy_gamut.min_luminance`
**Controls:** Darkest black achievable with CMY (L* value in LAB color space).

**LAB L* Scale:**
- `0` = Pure black
- `50` = Medium gray
- `100` = Pure white

**Typical Values:**

| Value     | CMY Capability                  | Hatching Behavior                  |
|-----------|---------------------------------|------------------------------------|
| `10-15`   | High-quality inkjet (rich blacks) | Minimal hatching, only deep blacks |
| `15-25`   | Airbrush with good coverage (current) | Moderate hatching for darkest regions |
| `25-35`   | Light/translucent inks          | Aggressive hatching for all shadows |

**Tuning:**
- **Lower** (10-15): Use if your CMY achieves deep, rich blacks → less hatching
- **Higher** (25-35): Use if your CMY produces light/muddy blacks → more hatching

---

#### `shadow_hatching.cmy_gamut.max_chroma`
**Controls:** Maximum color saturation CMY can achieve (chroma in LAB color space).

**LAB Chroma Scale:**
- `0` = Pure gray (no color)
- `50` = Moderate saturation
- `100+` = Highly saturated pure colors

**Typical Values:**

| Value     | CMY Capability                  | Hatching Behavior                  |
|-----------|---------------------------------|------------------------------------|
| `60-80`   | Standard CMY airbrush (current) | Hatch highly saturated colors      |
| `80-100`  | High-saturation inks            | Minimal hatching, wide gamut       |
| `40-60`   | Muted/pastel palette            | Aggressive hatching, narrow gamut  |

**Tuning:**
- **Lower** (40-60): Use if your CMY produces muted colors → more hatching
- **Higher** (80-100): Use if your CMY achieves vibrant colors → less hatching

---

#### `shadow_hatching.cmy_gamut.hue_ranges`
**Controls:** Hue ranges (in degrees, 0-360) that CMY can reproduce well.

**LAB Hue Scale:**
- `0°` = Red
- `60°` = Yellow
- `120°` = Green
- `180°` = Cyan
- `240°` = Blue
- `300°` = Magenta

**Examples:**

```yaml
# All hues (no filtering) - default
hue_ranges:
  - [0, 360]

# Exclude deep blues/purples (common CMY weakness)
hue_ranges:
  - [0, 210]    # Red → Cyan
  - [270, 360]  # Magenta → Red

# Warm colors only (reds, yellows, oranges)
hue_ranges:
  - [0, 90]
```

**Recommendation:** Leave at `[[0, 360]]` until you have calibration data.

---

### Darkness Levels

#### Structure
Each darkness level defines a **luminance range** and **hatching strategy**.

```yaml
darkness_levels:
  - l_min: 0.0    # Luminance range start (inclusive)
    l_max: 30.0   # Luminance range end (exclusive)
    passes: 1     # Number of hatching passes
    hatch_angles: [45]  # Hatch angles in degrees
```

**CRITICAL:** Ranges are **EXCLUSIVE** to prevent overlapping/double-tracing.

---

#### `passes`
**Controls:** Number of hatching passes for this darkness level.

| Value | Effect                                     | Density           |
|-------|--------------------------------------------|-------------------|
| `1`   | Single-direction hatching (current)        | Sparse            |
| `2`   | Double-pass (denser)                       | Medium-dense      |
| `3`   | Triple-pass (very dense, may over-ink)     | Dense             |

**Recommendation:** Start with `passes: 1` for all levels. Increase only for very dark regions if needed.

---

#### `hatch_angles`
**Controls:** Angles (in degrees) for hatching lines.

| Configuration         | Effect                                      | Density           |
|-----------------------|---------------------------------------------|-------------------|
| `[45]`                | Single direction (current)                  | Sparse            |
| `[45, 135]`           | Cross-hatch (perpendicular lines)           | Medium-dense      |
| `[0, 60, 120]`        | Triple hatch (60° apart)                    | Dense             |
| `[30, 90, 150]`       | Triple hatch (alternative angles)           | Dense             |

**Visual Impact:**
- **Single direction:** Stylized, directional shading
- **Cross-hatch:** Classic illustration style, balanced
- **Triple hatch:** Very dense, textured appearance

---

#### Example Configurations

**Configuration 1: Current (Sparse, Single-Direction)**
```yaml
darkness_levels:
  - l_min: 0.0
    l_max: 30.0
    passes: 1
    hatch_angles: [45]
  - l_min: 30.0
    l_max: 50.0
    passes: 1
    hatch_angles: [45]
  - l_min: 50.0
    l_max: 70.0
    passes: 1
    hatch_angles: [45]
```
**Result:** ~1,500-2,500 hatch paths, 8-12% coverage

---

**Configuration 2: Dense Cross-Hatch**
```yaml
darkness_levels:
  - l_min: 0.0
    l_max: 30.0
    passes: 2
    hatch_angles: [45, 135]
  - l_min: 30.0
    l_max: 50.0
    passes: 1
    hatch_angles: [45, 135]
  - l_min: 50.0
    l_max: 70.0
    passes: 1
    hatch_angles: [45]
```
**Result:** ~3,000-5,000 hatch paths, 20-30% coverage

---

**Configuration 3: Minimalist (Dark Regions Only)**
```yaml
darkness_levels:
  - l_min: 0.0
    l_max: 25.0
    passes: 1
    hatch_angles: [45]
```
**Result:** ~500-1,000 hatch paths, 3-5% coverage

---

### Hatching Density Controls

#### `shadow_hatching.spacing_scale`
**Controls:** Multiplier for hatch line spacing.

**Formula:** `line_spacing = pen_width × spacing_scale`

| Value      | Density             | Visual Effect                           |
|------------|---------------------|-----------------------------------------|
| `1.5-2.0`  | Dense               | Dark, textured, ink-heavy               |
| `2.5-3.0`  | Balanced (current: `2.5`) | Light hatching, colors show through     |
| `3.5-5.0`  | Very sparse         | Subtle accent, minimal ink              |

**Trade-off:**
- Higher → lighter appearance, more color visible, faster execution
- Lower → darker appearance, more ink, longer execution time

---

#### `shadow_hatching.min_line_spacing_mm`
**Controls:** Absolute minimum spacing between parallel hatch lines (in mm).

**Purpose:** Safety limit to prevent over-inking, regardless of `spacing_scale`.

| Value       | Effect                                    |
|-------------|-------------------------------------------|
| `0.3-0.4`   | Very dense (for fine 0.2-0.3mm pens)      |
| `0.5-0.6`   | Balanced (current: `0.5`, for 0.3mm pen)  |
| `0.8-1.0`   | Sparse (for thick 0.5-0.8mm pens)         |

**Recommendation:** Set to `1.5-2× pen_width` for balanced results.

---

#### `shadow_hatching.max_hatch_coverage`
**Controls:** Maximum fraction of canvas (0.0-1.0) that hatching can cover.

**Purpose:** Hard limit to prevent over-inking and ensure CMY colors dominate.

| Value        | Style                                     | Typical Use              |
|--------------|-------------------------------------------|--------------------------|
| `0.10-0.20`  | Sparse, pen as accent (current: `0.20`)   | Colorful illustrations   |
| `0.30-0.50`  | Moderate, pen and color balanced          | Mixed media style        |
| `0.60-0.75`  | Dense, pen-dominant                       | Pen-and-ink illustration |

**Algorithm Behavior:**
- Processing starts with darkest L* levels
- Once coverage limit is reached, no more hatching is added
- Lighter L* levels may be skipped entirely

**Example:**
```yaml
max_hatch_coverage: 0.20  # 20% max
```
If L* 0-30 hatching reaches 18% coverage, only 2% is available for L* 30-50 and L* 50-70.

---

### Morphology & Filtering

#### `shadow_hatching.min_area_px`
**Controls:** Minimum shadow region size (in pixels). Smaller regions are discarded.

**Purpose:** Remove noise and tiny shadow fragments.

| Value        | Effect                                    |
|--------------|-------------------------------------------|
| `100-300`    | Keep fine details, many small regions     |
| `500-1000`   | Balanced (current: `500`)                 |
| `1500-2000`  | Only large, significant shadow areas      |

**Recommendation:** Increase if you see excessive hatching in noisy/textured regions.

---

## Calibration

### `calibration.calibration_file`
**Controls:** Path to measured CMY gamut data (YAML file from calibration pipeline).

**Usage:**
1. Run calibration tests (see `src/data_pipeline/calibrate.py`)
2. Set path to generated calibration file
3. Pen tracer will use measured gamut instead of fallback values

**Example:**
```yaml
calibration_file: "outputs/calibration/cmy_gamut_2025-11-01.yaml"
```

**When to use:**
- **null:** Default fallback values (safe for testing)
- **Path:** After running calibration (production use)

---

### `calibration.margin`
**Controls:** Gamut expansion margin (safety factor, 0.0-1.0).

**Purpose:** Expand CMY gamut slightly to avoid harsh boundaries at gamut edges.

| Value   | Effect                                      |
|---------|---------------------------------------------|
| `0.00`  | Strict gamut (may have visible boundaries)  |
| `0.05`  | 5% expansion (current, recommended)         |
| `0.10+` | Conservative (more hatching, less reliance on CMY) |

**Recommendation:** Leave at `0.05` unless you see artifacts at color boundaries.

---

## Common Tuning Scenarios

### Scenario 1: Minimalist Line Art (Few Outlines, No Hatching)
**Goal:** Clean silhouettes only, no shadow fill.

```yaml
output:
  target_height_px: 3508
edge_detection:
  canny_low: 150.0
  canny_high: 300.0
  min_length_px: 30
shadow_hatching:
  enabled: false
```

**Result:** ~300-800 edge paths, 0% hatching

---

### Scenario 2: Watercolor-Style (Detailed Outlines, Light Hatching)
**Goal:** Capture soft gradients, minimal shadow fill.

```yaml
output:
  target_height_px: 2339  # Faster processing
edge_detection:
  canny_low: 50.0
  canny_high: 120.0
  min_length_px: 10
shadow_hatching:
  spacing_scale: 4.0
  max_hatch_coverage: 0.10
  darkness_levels:
    - l_min: 0.0
      l_max: 20.0
      passes: 1
      hatch_angles: [45]
```

**Result:** ~2,000-4,000 edge paths, ~500-1,000 hatch paths

---

### Scenario 3: Dense Cross-Hatch Illustration
**Goal:** Traditional pen-and-ink style, heavy hatching.

```yaml
output:
  target_height_px: 3508
edge_detection:
  canny_low: 80.0
  canny_high: 180.0
shadow_hatching:
  spacing_scale: 2.0
  max_hatch_coverage: 0.50
  darkness_levels:
    - l_min: 0.0
      l_max: 30.0
      passes: 2
      hatch_angles: [45, 135]
    - l_min: 30.0
      l_max: 50.0
      passes: 1
      hatch_angles: [45, 135]
    - l_min: 50.0
      l_max: 70.0
      passes: 1
      hatch_angles: [45]
```

**Result:** ~1,500-2,500 edge paths, ~4,000-8,000 hatch paths

---

### Scenario 4: Fast Preview (Low Resolution, Sparse)
**Goal:** Quick iteration, draft quality.

```yaml
output:
  target_height_px: 1170  # ~100 DPI
edge_detection:
  canny_low: 100.0
  canny_high: 200.0
  min_length_px: 15
  simplify_tol_px: 3.0
shadow_hatching:
  spacing_scale: 3.5
  max_hatch_coverage: 0.15
```

**Result:** ~300-800 total paths, <1 second processing

---

## Troubleshooting

### Problem: "Too many outlines, image looks busy"
**Solution:** Increase edge detection thresholds
```yaml
canny_low: 120.0   # Was: 100.0
canny_high: 250.0  # Was: 200.0
min_length_px: 30  # Was: 20
```

---

### Problem: "Missing important edges/details"
**Solution:** Lower edge detection thresholds
```yaml
canny_low: 70.0    # Was: 100.0
canny_high: 150.0  # Was: 200.0
min_length_px: 10  # Was: 20
```

---

### Problem: "Hatching is too dense, covers too much"
**Solution:** Increase spacing, reduce coverage
```yaml
spacing_scale: 3.5         # Was: 2.5
max_hatch_coverage: 0.10   # Was: 0.20
```

---

### Problem: "Not enough hatching, dark regions look weak"
**Solution:** Decrease spacing, increase coverage
```yaml
spacing_scale: 2.0         # Was: 2.5
max_hatch_coverage: 0.35   # Was: 0.20
```

---

### Problem: "Double-tracing (overlapping hatching patterns)"
**Solution:** Check darkness levels are EXCLUSIVE
```yaml
# BAD (overlapping ranges):
- l_min: 0.0
  l_max: 50.0
- l_min: 30.0
  l_max: 70.0

# GOOD (exclusive ranges):
- l_min: 0.0
  l_max: 30.0
- l_min: 30.0
  l_max: 50.0
- l_min: 50.0
  l_max: 70.0
```

---

### Problem: "Hatching appears in colored regions (not just shadows)"
**Solution:** Verify gamut-aware mode is enabled and check gamut parameters
```yaml
gamut_aware: true  # Must be true
cmy_gamut:
  min_luminance: 15.0  # Adjust based on your CMY capability
  max_chroma: 80.0
```

---

### Problem: "Processing is too slow"
**Solution:** Reduce resolution, increase simplification
```yaml
output:
  target_height_px: 2339  # Was: 3508
edge_detection:
  simplify_tol_px: 3.0    # Was: 1.5
```

---

### Problem: "G-code file is too large"
**Solution:** Increase simplification tolerance
```yaml
simplify_tol_px: 3.0  # Was: 1.5
```

---

### Problem: "Output resolution doesn't match expected"
**Solution:** Check min/max bounds aren't clamping
```yaml
output:
  min_px: 512   # Lower if needed
  max_px: 4096  # Raise if needed (but watch memory)
```

---

## Additional Resources

- **Implementation:** `src/data_pipeline/pen_tracer.py`
- **Schema:** `configs/schema/pen_tracer.v2.yaml`
- **Unit Tests:** `tests/test_pen_tracer.py`
- **Quick Start:** `documentation/PEN_TRACER_QUICKSTART.md`
- **Implementation Details:** `documentation/PEN_TRACER_IMPLEMENTATION.md`

---

**Questions or issues?** Check the logs for detailed processing stats, or enable `debug.save_intermediates: true` to inspect intermediate artifacts.

