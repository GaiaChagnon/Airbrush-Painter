# CPU Reference Renderer: OpenCV Distance-Transform Model

## Overview

The CPU reference renderer has been completely rewritten to use an **OpenCV distance-transform** approach that produces **realistically visible strokes** with physically accurate behavior. This replaces the previous Gaussian splat model which produced nearly invisible strokes due to normalization issues.

## Key Changes

### 1. **Physical Model: Flat Core + Gaussian Skirt**

Instead of normalized Gaussian stamps, the new model:

1. **Rasterizes a 1-pixel centerline** using OpenCV polyline drawing
2. **Computes distance transform** to get radial distance from centerline
3. **Builds radial opacity profile**:
   - **Flat core**: α(r) = 1.0 for r ≤ r_core
   - **Gaussian skirt**: α(r) = exp(-((r - r_core)/σ)^p) for r > r_core
   - **Hard cutoff** at r_max to avoid infinite tails
4. **Mass-per-mm scaling** (not normalized per-sample)
5. **Speed-aware width and deposition**

This matches real airbrush physics: dense core under the nozzle with soft falloff.

### 2. **Configuration-Driven Physics**

All physics parameters now live in **`configs/sim/renderer_cpu.v1.yaml`**:

```yaml
# Width model: z → [width_min, width_max], with speed scaling
width_model:
  z_knots_mm: [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
  width_min_mm: [0.8, 1.0, 1.3, 1.6, 2.0, 2.8, 3.5]
  width_max_mm: [2.2, 2.8, 3.6, 4.5, 5.5, 7.0, 8.5]
  v_knots_mm_s: [10.0, 30.0, 60.0, 120.0, 240.0]
  width_scale: [1.20, 1.10, 1.00, 0.88, 0.75]  # faster → narrower

# Deposition model: mass per second vs Z, scaled by speed
deposition:
  z_knots_mm: [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
  mass_per_sec: [0.65, 0.58, 0.50, 0.43, 0.37, 0.28, 0.22]
  speed_exponent: 1.0  # mass_per_mm = mass_per_sec / v^exp
  k_mass: 2.5          # global amplitude scale

# Profile shape
profile:
  core_frac: 0.40         # r_core = 0.40 * (width/2)
  skirt_sigma_frac: 0.28  # σ_skirt = 0.28 * (width/2)
  skirt_power: 1.8        # Gaussian power (higher = sharper falloff)
  margin_factor: 1.5      # ROI margin multiplier
```

**Calibration will rewrite these values** based on real measurements.

### 3. **Visibility Gates**

Strokes that are too faint are **automatically skipped** to save compute and avoid noise:

```yaml
visibility:
  min_alpha_visible: 0.0125        # zero out pixels below this
  min_delta_e_visible: 0.8         # perceptual threshold (ΔE2000)
  min_stroke_coverage: 0.0001      # skip if coverage < 0.01%
  min_center_luminance_drop: 0.05  # center must darken by ≥5%
```

### 4. **Speed & Width Scaling**

- **Width**: Higher Z → wider spray; higher speed → narrower (aerodynamic focusing)
- **Mass per mm**: Inversely proportional to speed (faster = less ink per length)
  - `mass_per_mm = mass_per_sec(z) / v^speed_exponent`
- **Step-invariant**: Total ink deposited is independent of sampling density

### 5. **Alpha-Over Compositing (Layering)**

Uses standard Porter-Duff "over" operator in linear RGB:

```python
C_out = C_in * (1 - α_stamp) + C_paint * α_stamp
α_out = α_in + α_stamp * (1 - α_in)
```

Yellow over green → yellowish-green (as expected).

**Future**: Kubelka-Munk scattering model can be enabled via `mixing.mode: "kubelka_munk"`.

## Usage

### Quick Preview

```bash
# Render a test stroke and inspect it visually
python scripts/preview_stroke_cpu.py --output preview.png

# Try different parameters
python scripts/preview_stroke_cpu.py --z 10.0 --speed 100.0 --color cyan
```

This will:
- Render a horizontal stroke
- Report coverage and center luminance drop
- Warn if stroke is nearly invisible
- Save PNG for visual inspection

### In Tests

```python
from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import fs

# Load configs
env_cfg = {
    'work_area_mm': (210.0, 297.0),
    'render_px': (908, 1280)
}
sim_cfg = {
    'renderer_cpu_config': 'configs/sim/renderer_cpu.v1.yaml'
}
luts = load_toy_luts(high_visibility=True)

# Initialize renderer
renderer = CPUReferenceRenderer(sim_cfg, env_cfg, luts)

# Render stroke
canvas = np.ones((908, 1280, 3), dtype=np.float32)
alpha = np.zeros((908, 1280), dtype=np.float32)

stroke_dict = {
    'id': 'test-00000',
    'bezier': {
        'p1': (50.0, 148.5),
        'p2': (80.0, 148.5),
        'p3': (130.0, 148.5),
        'p4': (160.0, 148.5)
    },
    'z_profile': {'z0': 6.0, 'z1': 6.0},
    'speed_profile': {'v0': 50.0, 'v1': 50.0},
    'color_cmy': {'c': 0.8, 'm': 0.2, 'y': 0.1}
}

canvas, alpha = renderer.render_stroke(canvas, alpha, stroke_dict)
```

## Test Suite

The test suite (`tests/test_cpu_golden.py`) includes:

### Golden Image Tests

- `test_golden_straight_line`: Horizontal line
- `test_golden_curved_stroke`: S-curve with varying z/v
- `test_golden_multi_stroke`: Three overlapping strokes (CMY layering)
- `test_golden_boundary_graze`: Stroke near canvas edge
- `test_golden_tiny_stroke`: Micro-stroke for anti-aliasing

**Metrics validated**:
- PSNR ≥ 28 dB
- SSIM ≥ 0.92
- ΔE2000 mean ≤ 2.0
- Coverage within expected range
- Center luminance drop ≥ 0.05

### Physics Tests

- `test_visibility_sanity`: Strokes are actually visible (not near-white)
- `test_width_control`: Width increases with Z as configured
- `test_speed_scaling`: Doubling speed reduces coverage (mass per mm halves)
- `test_determinism`: Rendering is deterministic across runs
- `test_layering_alpha_over`: Yellow over green → yellowish-green
- `test_coverage_stability`: Coverage metrics stable across runs

### Run Tests

```bash
# All golden tests
pytest tests/test_cpu_golden.py -m golden -v

# All physics tests
pytest tests/test_cpu_golden.py -m physics -v

# Regenerate golden images (manual, not in CI)
pytest tests/test_cpu_golden.py --regen-golden
```

## Configuration Reference

### Full Schema (`renderer_cpu.v1.yaml`)

```yaml
schema: renderer_cpu.v1
mode: "opencv_distance"  # or "stamp_train" (future)

visibility:
  min_alpha_visible: 0.0125
  min_delta_e_visible: 0.8
  min_stroke_coverage: 0.0001
  min_center_luminance_drop: 0.05

width_model:
  z_knots_mm: [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
  width_min_mm: [0.8, 1.0, 1.3, 1.6, 2.0, 2.8, 3.5]
  width_max_mm: [2.2, 2.8, 3.6, 4.5, 5.5, 7.0, 8.5]
  v_knots_mm_s: [10.0, 30.0, 60.0, 120.0, 240.0]
  width_scale: [1.20, 1.10, 1.00, 0.88, 0.75]

profile:
  type: "flat_core_gaussian_skirt"
  core_frac: 0.40
  skirt_sigma_frac: 0.28
  skirt_power: 1.8
  margin_factor: 1.5

deposition:
  z_knots_mm: [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
  mass_per_sec: [0.65, 0.58, 0.50, 0.43, 0.37, 0.28, 0.22]
  speed_exponent: 1.0
  k_mass: 2.5

mixing:
  mode: "layer_over"  # or "kubelka_munk"
  km_params_path: null

randomness:
  seed: 42
  speckle: true
  speckle_gain: 0.08
  speckle_scale: 2.0

sampling:
  max_step_mm: 0.25
  min_samples: 8
```

### Calibration Integration

The calibration script (`src/data_pipeline/calibrate.py`) will:

1. **Measure widths** from PSF/alpha measurements across Z and V
2. **Fit width curves**: `width_min_mm(z)`, `width_max_mm(z)`, `width_scale(v)`
3. **Estimate mass per second**: From dot tests (spray time vs measured alpha)
4. **Tune `k_mass`**: Via layering tests (predicted vs measured overlap color)
5. **Write** `renderer_cpu.v1.yaml` with calibrated values

## Algorithm Details

### Distance-Transform Profile

```python
# 1. Rasterize 1-px centerline
mask = np.zeros((H, W), np.uint8)
cv2.polylines(mask, [polyline_px], isClosed=False, color=255, thickness=1)

# 2. Distance transform (pixels from centerline)
dist_px = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

# 3. Convert to mm
dist_mm = dist_px / dpi_avg

# 4. Build radial profile
r_core = core_frac * 0.5 * width_mm
sigma_skirt = skirt_sigma_frac * 0.5 * width_mm

phi = np.ones_like(dist_mm)
skirt_mask = dist_mm > r_core
phi[skirt_mask] = np.exp(-((dist_mm[skirt_mask] - r_core) / sigma_skirt) ** power)
phi[dist_mm > 0.5 * width_mm * margin_factor] = 0.0

# 5. Apply mass scaling
k_mass = 2.5  # from config
mass_per_mm = mass_per_sec(z) / v^speed_exponent
alpha_profile = np.clip(k_mass * mass_per_mm * phi, 0.0, 1.0)

# 6. Zero out imperceptible values
alpha_profile[alpha_profile < min_alpha_visible] = 0.0
```

### Why This Works

1. **No normalization**: Kernel amplitude is proportional to ink mass, not probability
2. **Mass conservation**: Total opacity = mass_per_mm × path_length (step-invariant)
3. **Flat core**: Center stays dark (like real spray under nozzle)
4. **Speed matters**: Faster → less ink per mm, narrower width
5. **Visibility gates**: Skip strokes that won't affect image (save compute)

## Differences from Old Model

| Aspect | Old (Gaussian Splat) | New (OpenCV Distance-Transform) |
|--------|---------------------|--------------------------------|
| **Profile** | Single normalized Gaussian | Flat core + Gaussian skirt |
| **Amplitude** | Normalized to sum=1 | Scaled by mass_per_mm |
| **Visibility** | Often invisible (too faint) | **Always visible** at center |
| **Width** | Fixed σ (or PSF LUT) | Z-dependent + speed scaling |
| **Speed** | Affects spacing only | Affects width & mass per mm |
| **Layering** | Alpha-over (same) | Alpha-over (same) |
| **Config** | Hardcoded or mixed | **Pure YAML** (calibration-driven) |

## Performance

- **Speed**: Comparable to old model (OpenCV distance transform is fast)
- **Memory**: ROI-based rendering (only compute affected region)
- **Determinism**: Fully deterministic (seeded noise for speckle)

## Future Extensions

### Stamp Train Mode

Alternative rendering mode using textured nozzle patterns:

```yaml
mode: "stamp_train"
stamp_train:
  patterns: ["configs/sim/patterns/nozzle_0.png"]
  unit_diam_mm: 1.0
  spacing_mm: 0.5
  jitter_mm: 0.08
  noise_gain: 0.12
```

Places textured stamps along path (grainy, realistic appearance).

### Kubelka-Munk Mixing

Physically-based light scattering model for layering:

```yaml
mixing:
  mode: "kubelka_munk"
  km_params_path: "configs/sim/km_params.yaml"
```

Requires calibrated K (absorption) and S (scattering) coefficients.

## Troubleshooting

### Strokes Still Too Faint

1. **Increase `k_mass`** in `deposition` section (try 5.0 or 10.0)
2. **Increase `mass_per_sec`** values
3. **Decrease `min_alpha_visible`** (allow more fringe pixels)
4. **Check width**: Should be ≥ 0.8 mm at typical Z

### Strokes Too Wide

1. **Decrease `width_min_mm` / `width_max_mm`**
2. **Decrease `margin_factor`** (tighten ROI)
3. **Increase `skirt_power`** (sharper falloff)

### Strokes Not Speed-Sensitive

1. **Check `speed_exponent`** (should be ≈ 1.0 for inverse proportional)
2. **Verify `width_scale`** array (faster → lower scale)

### Coverage Unstable

1. **Check determinism test** (should pass)
2. **Verify `randomness.seed`** is fixed
3. **Disable speckle** temporarily (`speckle: false`)

## References

- **OpenCV Distance Transform**: [cv2.distanceTransform](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042)
- **Porter-Duff Alpha Compositing**: [PDF](https://graphics.pixar.com/library/Compositing/paper.pdf)
- **Kubelka-Munk Theory**: Applied later for scattering-based mixing

## Summary

The new CPU renderer produces **actually visible strokes** by:

1. Using a **flat core + Gaussian skirt** profile (matches real physics)
2. **Mass-per-mm scaling** (not normalized Gaussians)
3. **Speed-aware width and deposition**
4. **Visibility gates** to skip imperceptible strokes
5. **YAML-driven physics** (ready for calibration)

**Result**: Strokes are dark, visible, and physically realistic. The center of each stroke is always significantly darker than paper white, matching real airbrush behavior.

