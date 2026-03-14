# CLAUDE.md — Project Context for AI Assistants

## What this project is

Robotic painting system: Klipper XYZ gantry + airbrush + pen tool + syringe pumps.
Physics-based CPU renderer, YAML-validated configs, robot control CLI, and (planned)
RL-based stroke planner.

## Repository layout

```
scripts/                   Thin CLI entry points (demo, preprocess, verify)
src/
  airbrush_simulator/      CPU renderer (OpenCV distance-transform) + GPU differentiable renderer
  data_pipeline/           Image preprocessing, calibration stubs
  utils/                   Lowest layer — shared primitives (see below)
robot_control/
  calibration/             Guided calibration routines
  configs/                 Hardware configs (machine.yaml, jobs.yaml, lineart.yaml)
  gcode/                   Job IR -> G-code
  hardware/                Klipper UDS client, job executor, pump control, interactive jog
  images/previews/         Lineart tracer preview output (machine-coordinate PNGs)
  job_ir/                  Intermediate representation for job actions
  scripts/cli/             Unified robot CLI (Rich + questionary)
  tests/                   Robot-control pytest suite (no hardware needed)
configs/
  schema/                  Canonical YAML schemas (stroke, job, machine, pen, LUTs)
  sim/                     Simulator configs (physics, renderer_cpu, manual calibration)
tests/                     Core pytest suite (renderer, utils, G-code VM, preprocessing)
ci/                        Golden tests & visual regression baselines
```

## Dependency flow (strict, one-way)

```
scripts/  →  src/{simulator, data_pipeline, gui*, rl_agent*, env*}  →  src/utils/
```

Peer modules at the middle layer must not import each other.
`src/utils/` must not import from upper layers. No cyclic imports.
`*` = planned but not yet implemented.

## src/utils/ — the shared foundation

Every cross-cutting primitive lives here. Never re-implement what exists in utils.
If a needed helper is missing, add it to utils first (with tests), then call it.

| Module            | Scope                                                        |
|-------------------|--------------------------------------------------------------|
| `color`           | sRGB/linear, RGB-to-Lab, deltaE2000                         |
| `compute`         | mm/px conversion, LPIPS normalization, tiling, bounds        |
| `geometry`        | Cubic Bezier eval/flatten, polyline length/bbox, clipping    |
| `strokes`         | Stroke IDs, vec/YAML round-trip, heatmap, spatial filter     |
| `validators`      | Pydantic schemas for all config types                        |
| `gcode_generator` | Stroke linearization, coordinate transforms, G-code output   |
| `gcode_vm`        | G-code virtual machine (time estimation, soft-limit checks)  |
| `fs`              | Atomic YAML/image writes, symlinks, dir utilities            |
| `logging_config`  | Structured logging (JSON/console). No `print()` anywhere.    |
| `metrics`         | PSNR, SSIM, coverage, edge-preservation                     |
| `hashing`         | SHA-256 for files/tensors (provenance)                       |
| `torch_utils`     | Seeding, channels-last, device/AMP helpers, model summary    |
| `profiler`        | Timer context manager, NVTX ranges                          |
| `mlflow_helpers`  | MLflow logging (graceful no-op when unavailable)             |

**Note:** cursor rules reference `io.*` but the module was renamed to `fs`.
Use `src.utils.fs` for all atomic I/O: `atomic_yaml_dump`, `load_yaml`,
`atomic_save_image`, `symlink_atomic`, `ensure_dir`.

## Key invariants

- **Geometry = millimeters** end-to-end. Convert at boundaries only via
  `compute.mm_to_px` / `compute.px_to_mm`.
- **Images = linear RGB [0, 1]** inside the pipeline. sRGB at I/O boundaries.
- **LPIPS = FP32 only**, inputs [-1, 1] via `compute.normalize_img_for_lpips`.
- **YAML-only** configs. No JSON. Validate via `src.utils.validators`.
- **Reward = LPIPS improvement only.** No time/ink penalties. Fixed stroke cap.
- **DGX Spark defaults:** BF16 + channels-last for networks; LUT tensors FP32.
- **Atomic writes** for all GUI-visible artifacts (`fs.atomic_*`, `fs.symlink_atomic`).
- **No `print()`** for runtime output. Use `logging_config.setup_logging()` +
  `logging.getLogger(__name__)`.

## Coordinate transforms (lineart tracer)

`PaperTransform` maps image coordinates (mm, top-left, Y-down) to machine
coordinates (mm, bottom-left, Y-up). A **180-degree pre-rotation** is applied
inside `image_to_machine()` to compensate for the physical bed axis
orientation: the machine's effective frame, when viewed from the operator's
position, is rotated 180 degrees from the nominal convention.

Preview images are rendered by `render_machine_preview()` in machine
coordinates, using a standardised colour palette:

| Element        | Colour (BGR)       | Hex       |
|----------------|--------------------|-----------|
| Paper border   | `(219, 152, 52)`   | `#3498DB` |
| Margin border  | `(34, 126, 230)`   | `#E67E22` |
| Drawing paths  | `(60, 60, 60)`     | `#3C3C3C` |
| Labels / text  | `(140, 130, 120)`  | `#788C8C` |

These constants (`_CLR_*`) are defined in `run_lineart_tracer.py` near the
preview section.

## Lineart tracer features

The lineart tracer supports four vectorization modes configured in
`robot_control/configs/lineart.yaml`:

- **line_tracing** -- Potrace + vpype; best for schematics and text.
- **hatched** -- hatched library; best for photos and portraits.
- **hatching** -- zone-bounded cross-hatching; best for maps and diagrams.
- **flow_imager** -- vpype-flow-imager streamlines; best for artistic renderings.

The **flow_imager** mode has two additional flags:

- `invert: true` -- inverts the image so light features (faces, white objects)
  receive dense streamlines. Essential for images with predominantly white
  subjects against dark backgrounds.
- `auto_contrast: true` -- applies CLAHE (Contrast Limited Adaptive Histogram
  Equalization) before processing. Spreads the brightness histogram so
  low-contrast images produce stronger density variation. Applied before
  invert.

The CLI tracer (`robot_control/scripts/cli/lineart_tracer.py`) uses Rich
panels for initialization output, a Rich progress bar with percentage for
drawing execution, and `questionary.checkbox` for parameter editing (select
parameters to modify, then edit chosen ones in a loop).

Preview images are saved to `robot_control/images/previews/`.

## CLI defaults: Rich + questionary

All terminal interfaces (robot_control/scripts/cli/) use:

- **`rich`** (>=13.7) for rendering: `Console`, `Panel`, `Table`, `Live`,
  `Progress`, `Text.from_markup()`. Status bars, probe grids, pump diagrams,
  keymap panels — all Rich renderables.
- **`questionary`** (>=2.0) for all prompts: `select`, `confirm`, `checkbox`,
  `text`, `path`, `press_any_key_to_continue`. Never raw `input()`.
- **`readchar`** (>=4.0) for single-keypress capture in interactive jog mode.
- **`plotext`** (>=5.2) for inline terminal charts (pump volume calibration).

When building any new CLI or interactive feature, use these libraries. Match
the existing pattern: clear screen, render persistent panels at top, then
questionary prompt below.

## Behavioral rules for AI assistants

### Library proposals

Before introducing a new external dependency to solve a task, **ask the user
first** using the questions/AskQuestion tool. Explain:
1. What the library does and why it fits better than building from scratch.
2. The alternative (implementing it manually) and the trade-off.

Only proceed after the user approves. This applies to non-trivial libraries,
not standard-library modules or deps already in `requirements.txt`.

### Planning protocol

When creating a plan for a multi-step task, **always ask clarifying questions**
before starting implementation. Surface:
- Ambiguities in the request.
- Design trade-offs with meaningful alternatives.
- Potential improvements the user didn't mention but should consider.

Use the AskQuestion tool to batch these into structured choices where possible.

### Code style

- Python >= 3.10, full type hints. Prefer `TypedDict` / `pydantic` for structured data.
- NumPy-style docstrings for complex functions; inline `#` for simple ones.
- Document shapes, dtypes, ranges, units (especially tensors and mm geometry).
- No star imports. No commented-out code. Lines <= 100 cols.
- `ruff` + `black` + `isort` formatting.
- Tests in `tests/` (core) or `robot_control/tests/` (hardware layer), named `test_*.py`.
- External vendor APIs (OpenCV, LPIPS, nvdiffrast, Potrace) wrapped behind our modules.

### Config changes

When adding config fields: update the YAML schema in `configs/schema/`, the
Pydantic model in `src/utils/validators.py`, and the test in `tests/`.

## Running the project

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CPU renderer demo
python scripts/demo_alcohol_ink.py

# Tests
python -m pytest tests/ -v
python -m pytest robot_control/tests/ -v

# Robot CLI (needs Klipper hardware)
python robot_control/scripts/robot_cli.py
```

Container: `Dockerfile` based on `nvcr.io/nvidia/pytorch:2.4.0-py3` (CUDA 12.4, aarch64).

## Non-negotiable rules

1. **Read the codebase first.** Examine existing patterns, utils, and architecture
   before writing anything. Search headers and imports to find reusable functions.
2. **No code duplication.** Reuse existing functions and constants. No `foo`,
   `foo_improved`, `foo_improved_v2` chains -- fix the original.
3. **No dead code.** If code isn't called, delete it. No commented-out blocks.
4. **Test every function.** Every public function gets a test. Tests must reflect
   real usage, be designed to reveal flaws, and produce verbose output for debugging.
   No cheater tests that just assert `True`.
5. **No magic numbers/strings.** Use named constants. Never hardcode status codes,
   paths, or thresholds inline.
6. **Specific error handling.** Catch specific exception types, propagate properly.
   No bare `except` or lazy `catch-log-ignore`.
7. **Match existing naming.** Read the codebase naming patterns before inventing new
   conventions. Consistent parameter order across similar functions.
8. **No over-engineering.** Simple functions over abstractions. Don't build a factory
   when a function call works.
9. **Separation of concerns.** No validation in handlers, no I/O in pure compute,
   no business logic in UI code.
10. **No resource leaks.** Close connections, clear timers, remove listeners, clean
    up file handles. Use context managers.
11. **Clean async.** No nested callback chains. Use `async/await` or break into
    smaller functions.
12. **Robot-related CLI work goes in `robot_control/scripts/cli/`.** Follow the
    existing pattern: `RobotApp` context, Rich panels, questionary prompts.
