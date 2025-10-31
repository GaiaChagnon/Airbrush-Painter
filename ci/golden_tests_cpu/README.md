# Golden Test Data - CPU Reference Renderer

This directory contains golden images and acceptance thresholds for the CPU reference renderer regression tests.

## Purpose

Golden tests validate that rendering behavior remains consistent across code changes. They catch unintended changes to:
- Stroke rendering (polyline flattening, splatting)
- LUT interpolation (color, alpha, PSF)
- Compositing (alpha-over, accumulation)
- Boundary handling (clipping, edge cases)

## Directory Structure

```
ci/golden_tests_cpu/
├── README.md              # This file
├── thresholds.yaml        # Acceptance criteria
├── straight_line.png      # Golden image: simple horizontal stroke
├── curved_stroke.png      # Golden image: S-curve with varying z/v
├── multi_stroke.png       # Golden image: three overlapping strokes
├── boundary_graze.png     # Golden image: stroke near canvas edge
└── tiny_stroke.png        # Golden image: micro-stroke for anti-aliasing
```

## Golden Test Cases

### 1. straight_line.png
**Purpose:** Baseline test for horizontal stroke rendering  
**Parameters:**
- Bezier: (50, 148.5) → (160, 148.5) [nearly straight]
- Z: 10 mm (constant)
- Speed: 50 mm/s (constant)
- Color: High cyan (C=0.8, M=0.2, Y=0.1)

**What it tests:**
- Basic splatting and PSF application
- Linear arc-length sampling
- Constant z/v interpolation

### 2. curved_stroke.png
**Purpose:** Geometric fidelity for curved strokes  
**Parameters:**
- Bezier: S-curve from (50,50) to (200,150)
- Z: 5 → 15 mm (varying height)
- Speed: 100 → 200 mm/s (varying speed)
- Color: High magenta (C=0.2, M=0.8, Y=0.3)

**What it tests:**
- Bezier flattening with adaptive subdivision
- Z/V interpolation along curve
- PSF width variation (σ depends on z,v)

### 3. multi_stroke.png
**Purpose:** Compositing and overlap handling  
**Parameters:**
- Three overlapping horizontal strokes
- Colors: Cyan, Magenta, Yellow (CMY primaries)
- Varying z (8, 10, 12 mm) and v (50, 60, 70 mm/s)

**What it tests:**
- Alpha-over compositing
- Alpha accumulation (clamping at 1.0)
- Color mixing via LUT

### 4. boundary_graze.png
**Purpose:** Boundary clipping and edge cases  
**Parameters:**
- Short stroke near top-left corner (5,5) → (20,20)
- Z: 8 → 12 mm
- Speed: 75 → 125 mm/s
- Color: Mid-gray (C=0.5, M=0.5, Y=0.5)

**What it tests:**
- Kernel clipping at canvas boundaries
- ROI extraction (y_min, y_max, x_min, x_max)
- No out-of-bounds writes

### 5. tiny_stroke.png
**Purpose:** Anti-aliasing and sub-pixel accuracy  
**Parameters:**
- Micro-stroke: (100,100) → (103,101.5) [~3 mm length]
- Z: 10 mm (constant)
- Speed: 50 mm/s (constant)
- Color: Low saturation (C=0.3, M=0.3, Y=0.3)

**What it tests:**
- Polyline sampling density (Δs ≤ 0.5·σ)
- Sub-pixel positioning
- Kernel normalization for small stamps

## Acceptance Thresholds

See `thresholds.yaml` for current values. Typical thresholds:
- **PSNR ≥ 28 dB**: Strict pixel-level match
- **SSIM ≥ 0.92**: High structural similarity
- **ΔE2000 ≤ 2.0**: Just-noticeable color difference
- **Coverage ±5%**: Painted area stability

## Regenerating Golden Images

Golden images are **versioned with the codebase**. Regenerate only after intentional rendering changes (e.g., LUT update, PSF model change, physics refinement).

### Step 1: Delete existing goldens
```bash
rm ci/golden_tests_cpu/*.png
```

### Step 2: Run tests (will generate new goldens)
```bash
pytest tests/test_cpu_golden.py -v
# Tests will SKIP with message "Generated golden image: ..."
```

### Step 3: Inspect new goldens
```bash
# Visually inspect generated PNGs
# Check that they match expected rendering behavior
```

### Step 4: Commit if correct
```bash
git add ci/golden_tests_cpu/*.png
git commit -m "Update golden images for <reason>"
# Example: "Update golden images for improved PSF kernel (v2.1)"
```

### Step 5: Run tests again (should pass)
```bash
pytest tests/test_cpu_golden.py -v
# All tests should PASS with metrics within thresholds
```

## CI Integration

Golden tests run automatically on:
- Every pull request (prevents regressions)
- Nightly builds (catches platform-specific drift)
- Release branches (gates production deploys)

### CI Workflow
```yaml
# .github/workflows/test.yml
- name: Run golden tests
  run: pytest tests/test_cpu_golden.py --tb=short
  
- name: Upload golden diffs on failure
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: golden-diffs
    path: ci/golden_tests_cpu/*_diff.png
```

## Troubleshooting

### Test fails with "PSNR too low"
**Cause:** Pixel-level mismatch (rendering change or platform difference)  
**Fix:**
1. Check if change was intentional (LUT update, kernel change)
2. If intentional: regenerate goldens (see above)
3. If unintentional: debug renderer (check polyline, splatting, interpolation)

### Test fails with "ΔE too high"
**Cause:** Color shift (LUT change or color space conversion bug)  
**Fix:**
1. Verify color LUT unchanged (hash check)
2. Check sRGB↔linear conversions
3. Validate trilinear interpolation on CMY cube

### Test fails with "Coverage difference"
**Cause:** Alpha accumulation change (kernel normalization or compositing bug)  
**Fix:**
1. Check kernel sum = 1.0
2. Verify alpha clamping logic (≤1.0)
3. Inspect alpha map visually

## Metrics Explained

### PSNR (Peak Signal-to-Noise Ratio)
- **Range:** 0 to ∞ dB (higher is better)
- **Interpretation:**
  - \> 30 dB: Excellent match (invisible differences)
  - 25–30 dB: Good match (minor artifacts)
  - < 25 dB: Poor match (visible differences)
- **Formula:** 10·log₁₀(1 / MSE)

### SSIM (Structural Similarity Index)
- **Range:** -1 to 1 (higher is better, 1 = identical)
- **Interpretation:**
  - \> 0.95: Excellent structural match
  - 0.90–0.95: Good structural match
  - < 0.90: Poor structural match
- **Measures:** Luminance, contrast, structure

### ΔE2000 (CIEDE2000 Color Difference)
- **Range:** 0 to ∞ (lower is better)
- **Interpretation:**
  - < 1.0: Imperceptible
  - 1.0–2.0: Just noticeable
  - 2.0–10.0: Noticeable
  - \> 10.0: Large difference
- **Standard:** CIE color science (perceptual uniformity)

### Coverage Fraction
- **Range:** 0 to 1 (fraction of painted pixels)
- **Formula:** mean(alpha_map > 0)
- **Use:** Detect mass loss/gain bugs

## Version History

- **2025-10-31**: Initial golden test suite (5 cases)
- Renderer version: v2.3.0
- LUT source: Toy LUTs (analytical models)
- Canvas: 908×1280 px (A4, ~4.3 px/mm)

## Notes

- Golden images use **toy LUTs** (uncalibrated analytical models)
- Once calibrated LUTs are available, regenerate all goldens
- Cross-platform: Tested on Ubuntu 22.04, CUDA 12.1 (CPU-only for goldens)
- Deterministic: Fixed seeds, no threading, FP32 only
