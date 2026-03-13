# Airbrush Painter

Robotic painting system built around a Klipper-controlled XYZ gantry with
airbrush, pen tool, and syringe-pump ink delivery. The project includes a
physics-based airbrush simulator (CPU renderer), a YAML-validated
configuration stack, and a full robot-control layer for executing drawings
on the physical machine.

---

## Repository structure

```
Airbrush-Painter/
├── robot_control/              Klipper-based gantry control (see robot_control/README.md)
│   ├── Arduino_Scripts/        Arduino firmware for servo control
│   ├── calibration/            Guided calibration routines
│   ├── configs/                Hardware configs (machine.yaml, jobs.yaml, lineart.yaml)
│   ├── gcode/                  Job IR -> G-code generation
│   ├── hardware/               Klipper UDS client, job executor, interactive jog
│   ├── job_ir/                 Intermediate representation for all job actions
│   ├── scripts/                CLI entry points (tracer, calibration, test scripts)
│   └── tests/                  pytest suite (runs without hardware)
│
├── src/                        Core library
│   ├── airbrush_simulator/     CPU reference renderer (OpenCV distance-transform model)
│   ├── data_pipeline/          Image preprocessing & calibration stubs
│   └── utils/                  Shared utilities (see below)
│
├── scripts/                    Top-level entry points
│   ├── demo_alcohol_ink.py     Renderer demo (CPU)
│   ├── preprocess_dataset.py   Preprocess target images
│   └── verify_preprocessed.py  Verify preprocessed images
│
├── configs/
│   ├── schema/                 YAML schemas (stroke, job, machine, pen, LUTs)
│   └── sim/                    Simulator configs (physics, renderer, manual calibration)
│
├── tests/                      Top-level pytest suite (renderer, utils, G-code)
├── ci/                         Golden tests & visual regression baselines (CPU renderer)
├── data/                       Raw images & calibration scans (.gitkeep placeholders)
├── outputs/                    Generated artifacts (.gitignored)
└── documentation/              Architecture docs & design notes
```

### `src/utils/` modules

| Module | Purpose |
|--------|---------|
| `color` | sRGB/linear conversion, RGB-to-Lab, deltaE2000 |
| `compute` | mm/px conversion, LPIPS normalization, tiling, bounds projection |
| `geometry` | Cubic Bezier evaluation/flattening, polyline length/bbox, clipping |
| `strokes` | Stroke ID generation, vec/YAML round-trip, heatmap, spatial filtering |
| `validators` | Pydantic schemas for stroke, machine, job, LUT, pen-tool, pen-tracer configs |
| `gcode_generator` | Stroke linearization, coordinate transforms, G-code file generation |
| `gcode_vm` | G-code virtual machine (time estimation, soft-limit violation detection) |
| `fs` | Atomic YAML/image writes, symlink helpers, directory utilities |
| `logging_config` | Structured JSON/console logging (no `print()`) |
| `metrics` | PSNR, SSIM, edge preservation, paint coverage |
| `hashing` | SHA-256 for files and tensors (provenance tracking) |
| `torch_utils` | Seeding, channels-last, device helpers, AMP context, model summary |
| `profiler` | Timer context manager, NVTX range wrapper |
| `mlflow_helpers` | MLflow parameter/artifact logging (graceful no-op when unavailable) |

---

## Prerequisites

- **Python 3.10+**
- **Hardware (for robot execution):** Klipper-controlled gantry with BTT Octopus
  board, Arduino for servo bridging, syringe pumps. See `robot_control/README.md`.
- **Hardware (for simulator only):** Any machine with a CPU; no GPU required.

## Installation

```bash
git clone <repo-url> && cd Airbrush-Painter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Some vectorization libraries need system packages:

```bash
# Debian / Ubuntu
sudo apt-get install libpotrace-dev libagg-dev
```

---

## Quick start

### 1. Run the CPU renderer demo

Renders a grid of airbrush strokes with varying colors, Z heights, and
speeds to `outputs/alcohol_ink_comprehensive_demo.png`.

```bash
python scripts/demo_alcohol_ink.py
```

### 2. Run the test suite

```bash
# Core library tests (renderer, utils, G-code)
python -m pytest tests/ -v

# Robot-control tests (no hardware needed)
python -m pytest robot_control/tests/ -v
```

### 3. Robot control

Full documentation in [`robot_control/README.md`](robot_control/README.md).

```bash
# Verify Klipper connectivity
python robot_control/scripts/test_connection.py

# Line-art tracing dry-run (B&W outlines, preview only)
python robot_control/scripts/run_lineart_tracer.py \
  --image "Syringe Pump Drawing v1.png" --dry-run --save-preview

# Execute on the robot
python robot_control/scripts/run_lineart_tracer.py \
  --image "Syringe Pump Drawing v1.png"

# Interactive keyboard jog controller
python robot_control/scripts/interactive_control.py

# Guided calibration (axes, endstops, servos, bed mesh)
python robot_control/scripts/calibrate.py --steps-x
```

---

## Configuration

All configuration is YAML-only, validated by Pydantic schemas in
`src/utils/validators.py`.

| Config area | Location | Description |
|-------------|----------|-------------|
| Schemas | `configs/schema/` | Canonical schemas for stroke, job, machine, pen, LUT |
| Simulator | `configs/sim/` | Physics model, CPU renderer parameters, manual calibration |
| Robot | `robot_control/configs/` | Machine dims, jobs, lineart tracer, Klipper printer.cfg |

### Key conventions

- **Geometry is in millimeters** end-to-end. Pixel conversions happen only at
  renderer/IO boundaries via `compute.mm_to_px` / `compute.px_to_mm`.
- **Images are linear RGB [0, 1]** inside the pipeline. sRGB conversion at IO only.
- **LPIPS** is always computed in FP32, inputs normalized to [-1, 1] via
  `compute.normalize_img_for_lpips`.

---

## CPU renderer

The reference renderer in `src/airbrush_simulator/cpu_reference.py` simulates
airbrush spray physics deterministically on CPU:

- Cubic Bezier to polyline with adaptive flattening
- Radial opacity profile: flat core + Gaussian skirt
- Speed-aware width and mass deposition
- CMY to linear RGB via trilinear color-LUT interpolation
- Transparent filter compositing (Beer-Lambert transmission model)
- Visibility gates to skip imperceptible strokes

Configured by `configs/sim/renderer_cpu.v1.yaml`.

---

## Hardware overview

| Component | Details |
|-----------|---------|
| **Gantry** | Dual-motor X, single Y, seesaw Z. GT2 belt, 0.9 deg steppers. |
| **Controller** | BTT Octopus (STM32H723) running Klipper |
| **Airbrush** | Iwata HP-CS, needle retract via Arduino servo |
| **Pen tool** | Fine-tip marker on spring-loaded Z |
| **Ink delivery** | 4x syringe pumps (C, M, Y, purge/IPA), 64-microstep lead-screw |
| **Workspace** | 450 x 320 x 80 mm |

See `robot_control/README.md` for wiring, calibration, and troubleshooting.

---

## Project status

**Working now:**
- CPU reference renderer with realistic spray physics
- Full utility stack (color science, geometry, G-code, validation, metrics)
- Robot control: line-art tracer (4 modes), interactive jog, calibration,
  syringe pump control
- Arduino servo firmware
- Test suites for renderer, utils, G-code VM, and robot-control layer

**Planned:**
- Calibrated LUTs from real scan data
- AI-based stroke planning (RL agent for CMY airbrush layer)
- Full CMY painting pipeline with ink delivery
