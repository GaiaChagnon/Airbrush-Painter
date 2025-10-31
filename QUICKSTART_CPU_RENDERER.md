# CPU Reference Renderer - Quick Start Guide

This guide will help you get the CPU reference renderer up and running in 5 minutes.

## Prerequisites

Ensure you have Python ≥3.10 and the required dependencies installed:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install numpy torch pillow matplotlib pyyaml pydantic lpips scikit-image
pip install pytest pytest-xdist  # For testing
```

## Step 1: Verify Installation

Test that the module imports correctly:

```bash
cd /workspace
python3 -c "from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts; print('✅ Import successful')"
```

## Step 2: Run Unit Tests

```bash
# Run all CPU renderer unit tests
pytest tests/test_cpu_renderer.py -v

# Run a subset (quick smoke test)
pytest tests/test_cpu_renderer.py::test_initialization -v
pytest tests/test_cpu_renderer.py::test_single_stamp_mass_preservation -v
```

**Expected output:** All tests should PASS with detailed metrics logged.

## Step 3: Generate Golden Images

On the first run, golden tests will generate reference images:

```bash
# Generate golden images (first run)
pytest tests/test_cpu_golden.py -v

# Expected: Tests SKIP with "Generated golden image: ..." messages

# Verify images exist
ls -lh ci/golden_tests_cpu/*.png

# Subsequent runs validate against goldens
pytest tests/test_cpu_golden.py -v

# Expected: Tests PASS with PSNR, SSIM, ΔE metrics
```

## Step 4: Preview a Stroke

Use the CLI tool to visualize a stroke:

```bash
# Simple horizontal red stroke
python3 scripts/preview_stroke_cpu.py \
    --inline \
    --x1 50 --y1 148.5 --x2 80 --y2 148.5 \
    --x3 130 --y3 148.5 --x4 160 --y4 148.5 \
    --z0 10 --z1 10 \
    --v0 50 --v1 50 \
    --c 0.8 --m 0.2 --y 0.1 \
    --output_dir outputs/preview_cpu

# View outputs
ls outputs/preview_cpu/
# Expected: stroke_render.png, alpha_map.png, psf_profiles.png, metadata.yaml
```

**Open the images:**
- `stroke_render.png`: Rendered stroke (sRGB, PNG)
- `alpha_map.png`: Coverage map (grayscale)
- `psf_profiles.png`: PSF width and alpha vs (z, v)

## Step 5: Render Multiple Strokes

Create a YAML file with multiple strokes:

```yaml
# strokes_example.yaml
schema: stroke.v1
strokes:
  - id: "test-00000-straight"
    bezier:
      p1: [50.0, 100.0]
      p2: [80.0, 100.0]
      p3: [130.0, 100.0]
      p4: [160.0, 100.0]
    z_profile:
      z0: 10.0
      z1: 10.0
    speed_profile:
      v0: 50.0
      v1: 50.0
    color_cmy:
      c: 0.8
      m: 0.2
      y: 0.1
      
  - id: "test-00001-curved"
    bezier:
      p1: [50.0, 150.0]
      p2: [100.0, 180.0]
      p3: [150.0, 170.0]
      p4: [200.0, 200.0]
    z_profile:
      z0: 5.0
      z1: 15.0
    speed_profile:
      v0: 100.0
      v1: 200.0
    color_cmy:
      c: 0.2
      m: 0.8
      y: 0.3
```

Render from file:

```bash
python3 scripts/preview_stroke_cpu.py \
    --stroke_file strokes_example.yaml \
    --output_dir outputs/multi_stroke
```

## Step 6: Programmatic Usage

Use the renderer in your Python code:

```python
from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
import numpy as np
from src.utils import color as color_utils

# 1. Setup renderer
env_cfg = {
    'work_area_mm': (210.0, 297.0),  # A4
    'render_px': (908, 1280)          # ~4.3 px/mm
}
luts = load_toy_luts()
renderer = CPUReferenceRenderer({}, env_cfg, luts)

# 2. Initialize canvas (white background)
canvas = np.ones((908, 1280, 3), dtype=np.float32)
alpha = np.zeros((908, 1280), dtype=np.float32)

# 3. Define stroke
stroke = {
    'id': 'example-001',
    'bezier': {
        'p1': (50.0, 100.0),
        'p2': (80.0, 100.0),
        'p3': (130.0, 100.0),
        'p4': (160.0, 100.0)
    },
    'z_profile': {'z0': 10.0, 'z1': 10.0},
    'speed_profile': {'v0': 50.0, 'v1': 50.0},
    'color_cmy': {'c': 0.8, 'm': 0.2, 'y': 0.1}
}

# 4. Render
canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)

# 5. Convert to sRGB and save
canvas_srgb = color_utils.linear_to_srgb(canvas)
canvas_uint8 = np.clip(canvas_srgb * 255, 0, 255).astype(np.uint8)

from PIL import Image
img = Image.fromarray(canvas_uint8, mode='RGB')
img.save('my_stroke.png')
```

## Common Tasks

### Validate Determinism

```python
# Render same stroke twice, verify identical output
canvas1, alpha1 = renderer.render_stroke(canvas.copy(), alpha.copy(), stroke)
canvas2, alpha2 = renderer.render_stroke(canvas.copy(), alpha.copy(), stroke)

assert np.allclose(canvas1, canvas2, atol=1e-7)
assert np.allclose(alpha1, alpha2, atol=1e-7)
print("✅ Deterministic")
```

### Measure Line Width

```python
# Take cross-section at mid-stroke
h, w = alpha.shape
cy = h // 2
profile = alpha[cy, :]

# Find FWHM
peak = profile.max()
half_max = 0.5 * peak
above_half = profile >= half_max
indices = np.where(above_half)[0]
fwhm_px = indices[-1] - indices[0]

print(f"Line width (FWHM): {fwhm_px:.2f} pixels")
```

### Inspect PSF and Alpha

```python
# Sample z, v points along stroke
z_samples = np.linspace(5.0, 15.0, 10)
v_samples = np.linspace(50.0, 200.0, 10)

for z, v in zip(z_samples, v_samples):
    sigma = renderer._interpolate_psf_lut(z, v)
    alpha_val = renderer._interpolate_alpha_lut(z, v)
    print(f"z={z:.1f}, v={v:.1f} → σ={sigma:.2f} px, α={alpha_val:.3f}")
```

## Troubleshooting

### Import Error: "No module named 'numpy'"

**Solution:** Install dependencies
```bash
pip install numpy torch pillow matplotlib pyyaml pydantic
```

### Test Failure: "PSNR too low"

**Cause:** Golden image mismatch (expected on first run)

**Solution:** Delete existing goldens and regenerate
```bash
rm ci/golden_tests_cpu/*.png
pytest tests/test_cpu_golden.py -v
```

### Preview Script Error: "Missing required parameters"

**Cause:** Inline mode requires all stroke parameters

**Solution:** Provide all required flags or use `--stroke_file`
```bash
# All required: x1-x4, y1-y4, z0, z1, v0, v1, c, m, y
python3 scripts/preview_stroke_cpu.py --inline \
    --x1 50 --y1 100 --x2 80 --y2 100 \
    --x3 130 --y3 100 --x4 160 --y4 100 \
    --z0 10 --z1 10 --v0 50 --v1 50 \
    --c 0.8 --m 0.2 --y 0.1 \
    --output_dir outputs/test
```

### Rendering Artifacts or Strange Colors

**Cause:** Toy LUTs are analytical approximations

**Solution:** Expected behavior until calibrated LUTs available. Verify:
- CMY values in [0,1]
- Z values in [0, 30] mm
- Speed values in [1, 300] mm/s

## Performance Tips

1. **Reduce canvas size for faster previews:**
   ```bash
   python3 scripts/preview_stroke_cpu.py ... --canvas_size 454,640
   ```

2. **Use simpler strokes for testing:**
   - Straight lines (horizontal/vertical) are fastest
   - Avoid very large σ (slow kernel operations)

3. **Profile rendering:**
   ```python
   import time
   start = time.time()
   canvas, alpha = renderer.render_stroke(canvas, alpha, stroke)
   elapsed = time.time() - start
   print(f"Rendered in {elapsed:.3f} seconds")
   ```

## Next Steps

1. ✅ **CPU renderer working** → You're here!
2. 🔄 **GPU renderer** → Implement differentiable CUDA path
3. 🔄 **Parity tests** → Validate CPU/GPU agreement
4. 🔄 **Real LUTs** → Calibrate from physical airbrush
5. 🔄 **Environment integration** → Use in RL training loop

## Resources

- **Full Documentation:** `CPU_RENDERER_IMPLEMENTATION.md`
- **Project Status:** `PROJECT_STATUS.md`
- **Global Plan:** `GlobalPlan.md`
- **Test Coverage:** `tests/test_cpu_renderer.py` (32+ tests)
- **Golden Tests:** `ci/golden_tests_cpu/README.md`

## Support

For issues or questions:
1. Check linter output: `python3 -m pylint src/airbrush_simulator/cpu_reference.py`
2. Run tests with verbose output: `pytest tests/test_cpu_renderer.py -vv`
3. Check logs in preview script: `--verbose` flag
4. Inspect intermediate outputs (alpha maps, PSF profiles)

---

**Status:** ✅ CPU Renderer Implementation Complete  
**Last Updated:** 2025-10-31  
**Version:** v1.0.0
