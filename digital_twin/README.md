# Digital Twin -- GPU Airbrush Simulator

Config-driven, differentiable, batch-capable stamp-splatting renderer
calibrated against real scanned measurements. Serves as the physics
backend for RL-based stroke planning.

## Quick start

```bash
# From repo root, with the venv active:
source .venv/bin/activate

# Run the test suite (CPU, no GPU required)
python -m pytest tests/test_digital_twin.py -v

# Launch the calibration wizard
python -m digital_twin.calibration_cli

# Use the simulator from code
python -c "
from digital_twin.gpu_simulator import GPUStampSimulator
sim = GPUStampSimulator('digital_twin/calibration.yaml')
canvas = sim.reset()
print('Canvas shape:', canvas.shape)
"
```

## Architecture

```
digital_twin/
  calibration.yaml      All physics parameters (YAML, no hardcoded constants)
  gpu_simulator.py      GPUStampSimulator -- the rendering engine
  calibration_cli.py    Block-based calibration wizard (Rich + questionary)
```

Supporting files elsewhere in the repo:

| File | Role |
|------|------|
| `src/utils/validators.py` | `CalibrationV1` Pydantic model + load/save |
| `tests/test_digital_twin.py` | 35 tests (schema, profile, compositing, LUT, sampling, batch, grad, determinism, round-trip) |

## Physical model

1. **Spray shape** -- soft cone with flat core + Gaussian skirt.
   Footprint radius `R(z)` from a 1-D LUT over nozzle height.
2. **Deposition** -- mass per mm of path `mass(z, v)` from a 2-D LUT
   over height and speed.
3. **Color** -- stable manifold `color_lut[C,M,Y] -> RGB` via trilinear
   interpolation on a 3-D CMY grid.
4. **Layering** -- multiplicative transmission compositing:
   `canvas_new = canvas_old * ((1-alpha) + alpha * paint_rgb)`.
   No wetness, drying, anisotropy, or order-dependent chemistry.

All parameters live in `calibration.yaml`. The simulator reads them at
init and converts to GPU tensors.

## Calibration workflow

The CLI wizard (`calibration_cli.py`) guides the operator through six
blocks, each targeting one physical question:

| Block | Fits | Input |
|-------|------|-------|
| 1. White reference | `paper_white_rgb` | Scanner measurement |
| 2. Dot sheet | `radius_lut_mm`, profile shape | Dot diameters + zoned RGB |
| 3. Line sheet | `mass_lut` | Line centre darkness at (z, speed) |
| 4. Color swatches | `color_lut` | Interior RGB per CMY recipe |
| 5. Layering | optional `layer_gain_lut` | Overlap-region RGB |
| 6. Preview/validate | validation summary | Simulates all measurements, reports delta-E |

Every measurement and fitted parameter is stored in `calibration.yaml`.

## Key invariants

- **Geometry = millimeters** end-to-end. Pixel conversion via DPI derived
  from `render.canvas_hw / render.work_area_mm`.
- **Images = linear RGB [0, 1]** inside the pipeline.
- **Canvas shape = `(B, 3, H, W)`** -- batched from day one for RL.
- **Differentiable** -- gradients flow through the compositor.
- **Deterministic** -- same config + same stroke = bitwise identical output.
- **FP32 throughout** -- LUTs and compositing stay in FP32.

## Configuration reference

See `calibration.yaml` for inline documentation of every field.
Schema validation is handled by `CalibrationV1` in `src/utils/validators.py`.

Key tunables:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k_mass` | 8.5 | Global opacity gain. Higher = darker strokes. |
| `profile.core_frac` | 0.40 | Flat-core radius fraction. Higher = wider opaque centre. |
| `profile.skirt_sigma_frac` | 0.28 | Skirt spread. Higher = softer edges. |
| `profile.skirt_power` | 1.8 | Tail shaping. Higher = sharper cutoff. |
| `profile.margin_factor` | 1.5 | ROI extent as multiple of R(z). |
| `sampling.max_step_mm` | 0.25 | Arc-length spacing between stamps. |

## Tests

```bash
python -m pytest tests/test_digital_twin.py -v
```

35 tests covering: schema validation (9), stamp profile (4),
compositing math (3), LUT interpolation (6), stroke sampling (4),
batch independence (1), differentiability (1), determinism (1),
calibration round-trip (2), preview tools (4).
