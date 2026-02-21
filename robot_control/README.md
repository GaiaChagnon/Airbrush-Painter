# robot_control

Motor control, calibration, and job execution for the Klipper-based pen/airbrush gantry.

Communicates with Klipper via its Unix Domain Socket (UDS) API using JSON + `0x03` (ETX) framing.  All hardware parameters live in `configs/machine.yaml` -- nothing is hardcoded in Python.

---

## Package structure

```
robot_control/
├── configs/                  Configuration layer
│   ├── machine.yaml          Single source of truth for all hardware params
│   ├── jobs.yaml             Paper size, speeds, Z heights for pen jobs
│   ├── lineart.yaml          Line-art tracer & hatched-fill configuration
│   ├── loader.py             YAML -> typed frozen dataclasses (MachineConfig)
│   └── printer_cfg.py        MachineConfig -> Klipper printer.cfg text
│
├── hardware/                 Klipper communication layer
│   ├── klipper_client.py     UDS client: connect, send G-code, query state
│   ├── job_executor.py       Runs Job IR via client (file or interactive mode)
│   └── interactive.py        Curses TUI for keyboard jog / manual control
│
├── images/                   Input images for run_lineart_tracer.py
│   └── previews/             Auto-generated dry-run previews
│
├── job_ir/                   Intermediate representation
│   └── operations.py         Frozen dataclasses for all job actions
│
├── gcode/                    Code generation
│   └── generator.py          Job IR -> G-code with coordinate transforms
│
├── calibration/              Calibration & test patterns
│   ├── patterns.py           Pattern generators returning Job IR operations
│   ├── routines.py           Guided interactive calibration sequences
│   └── measurement.py        User prompts and correction math
│
├── scripts/                  CLI entry points
│   ├── run_lineart_tracer.py Standalone line-art / hatched-fill tracer + robot execution
│   ├── run_tracer.py         Execute pre-generated pen_vectors.yaml on the robot
│   ├── test_motors.py        Hardware bring-up (motors, endstops, homing, circles)
│   ├── test_connection.py    Verify Klipper API reachable
│   ├── test_motion.py        Homing + coordinated XY move checks
│   ├── calibrate.py          Guided calibration entry point
│   ├── run_job.py            Run patterns or job files
│   └── interactive_control.py  Launch keyboard jog controller
│
└── tests/                    pytest suite (runs without hardware)
    ├── test_config_and_printer_cfg.py
    ├── test_gcode_generator.py
    ├── test_job_ir.py
    ├── test_klipper_client.py
    └── test_calibration.py
```

---

## Line-art tracer (`run_lineart_tracer.py`)

Standalone single-file script that vectorizes images and draws them on the robot.  Four operating modes:

| Mode | Purpose | Engine |
|------|---------|--------|
| `line_tracing` | B&W outlines for schematics, text, line drawings | Potrace + vpype |
| `hatched` | Colour hatching -- density-based fill good for photos | hatched library + vpype |
| `hatching` | Zone-bounded cross-hatching with auto-detected gray levels | Custom scanline engine + vpype |
| `flow_imager` | Flow-field streamlines for artistic rendering | vpype-flow-imager |

### Quick start

```bash
# Copy an image into the images folder
cp "data/raw_images/hard/Syringe Pump Drawing v1.png" robot_control/images/

# List available images
.venv/bin/python robot_control/scripts/run_lineart_tracer.py --list

# Line-tracing dry-run (B&W outlines, no robot)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --image "Syringe Pump Drawing v1.png" --dry-run --save-preview

# Line-tracing with higher threshold (for maps with light-gray features)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --image "peakpx (4).jpg" --threshold 180 --dry-run --save-preview

# Hatched mode dry-run (density gradients)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatched --image-path "data/raw_images/hard/peakpx (2).jpg" \
  --dry-run --save-preview

# Hatched with 8 gradient levels and custom pitch
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatched --image-path "data/raw_images/hard/peakpx.jpg" \
  --levels 32 64 96 128 160 192 224 --hatch-pitch 0.5 \
  --dry-run --save-preview

# Hatched without outline overlay
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatched --image-path "data/raw_images/hard/peakpx.jpg" \
  --no-outlines --dry-run --save-preview

# Hatching mode -- zone-bounded cross-hatching (maps, diagrams)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatching --image-path "robot_control/images/peakpx (4).jpg" \
  --hatching-angles 45 -45 --dry-run

# Hatching with 5 zones and single-direction hatching
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatching --image-path "robot_control/images/peakpx (4).jpg" \
  --n-zones 5 --hatching-angles 45 --dry-run

# Triple cross-hatching (0, 60, 120 degrees)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode hatching --image-path "robot_control/images/peakpx (4).jpg" \
  --hatching-angles 0 60 120 --dry-run

# Flow-imager (artistic streamlines)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode flow_imager --image-path "data/raw_images/hard/peakpx (2).jpg" \
  --dry-run --save-preview

# Flow-imager with hexagonal pattern and edge following
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode flow_imager --image-path "data/raw_images/hard/peakpx.jpg" \
  --n-fields 6 --edge-field-mult 1.0 --dry-run --save-preview

# Flow-imager with curl noise and dark-area swirling
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --mode flow_imager --image-path "data/raw_images/hard/peakpx.jpg" \
  --field-type curl_noise --dark-field-mult 1.0 --dry-run --save-preview

# Execute on the robot (any mode)
.venv/bin/python robot_control/scripts/run_lineart_tracer.py \
  --image "Syringe Pump Drawing v1.png"
```

### Configuration

The tracer reads three config files:

| File | Purpose |
|------|---------|
| `configs/machine.yaml` | Workspace dimensions, soft limits, connection |
| `configs/jobs.yaml` | Paper size/origin, pen speeds, Z heights |
| `configs/lineart.yaml` | Mode selection, tracing and hatching parameters |

CLI arguments override config values.  `lineart.yaml` is the primary config for tuning the vectorization pipeline.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| **Image** | | |
| `--image NAME` | -- | Image filename in `robot_control/images/` |
| `--image-path PATH` | -- | Direct path to any image (overrides `--image`) |
| `--list` | -- | List available images and exit |
| **Mode** | | |
| `--mode` | `line_tracing` | `line_tracing`, `hatched`, `hatching`, or `flow_imager` |
| **Line-tracing params** | | |
| `--merge-tolerance` | 5.0 | Max pixel distance for vpype endpoint merging |
| `--turdsize` | 10 | Potrace speckle filter (discard contours < N px^2) |
| `--threshold` | 128 | Binary threshold 0-255 |
| `--simplify-tol` | 0.05 | Douglas-Peucker tolerance (mm) |
| `--min-path-mm` | 0.5 | Drop paths shorter than this (mm) |
| **Hatched params** | | |
| `--levels N...` | 64 128 192 | Grayscale thresholds dividing density zones |
| `--hatch-pitch` | 0.6 | Densest hatching pitch (mm) |
| `--hatch-angle` | 45 | Hatching angle (degrees) |
| `--blur-radius` | 10 | Blur before thresholding |
| `--image-scale` | 1.0 | Scale factor before hatching |
| `--circular` | off | Concentric circles instead of diagonals |
| `--invert` | off | Invert image before processing |
| `--no-outlines` | off | Disable potrace outline overlay |
| **Hatching params** | | |
| `--n-zones` | 4 | Number of gray zones (incl. white).  Auto-detected from histogram |
| `--hatching-angles N...` | 45 | Hatch angles (deg).  Multiple = cross-hatch (e.g., `45 -45`) |
| `--hatching-blur` | 5 | Gaussian blur before zone detection |
| `--hatching-scale` | 1.0 | Image scale factor (reduce for speed) |
| `--hatching-no-outlines` | off | Disable potrace outline overlay |
| `--connect-gap` | 5.0 | Max pixel gap for zigzag line connection |
| **Flow-imager params** | | |
| `--noise-coeff` | 0.001 | Simplex noise frequency (smaller = smoother) |
| `--n-fields` | 1 | Rotated field copies (1=smooth, 6=hexagonal) |
| `--flow-min-sep` | 0.8 | Min flowline separation (px) |
| `--flow-max-sep` | 10.0 | Max flowline separation (px) |
| `--flow-min-length` | 0.0 | Min flowline length (px) |
| `--flow-max-length` | 40.0 | Max flowline length (px) |
| `--flow-max-size` | 800 | Resize image largest side (px) |
| `--flow-seed` | 42 | PRNG seed for flow field |
| `--field-type` | noise | `noise` or `curl_noise` |
| `--edge-field-mult` | off | Blend edge-following field (try 1.0) |
| `--dark-field-mult` | off | Blend dark-curling field (try 1.0) |
| **Z-axis** | | |
| `--z-contact` | from jobs.yaml | Z position where pen touches paper (mm) |
| `--z-retract` | from jobs.yaml | Retract distance above contact (mm) |
| **Speeds** | | |
| `--draw-speed` | from jobs.yaml | Drawing speed (mm/s) |
| `--travel-speed` | from jobs.yaml | Travel speed (mm/s) |
| **Paper** | | |
| `--paper-size W H` | from jobs.yaml | Paper dimensions (mm) |
| `--paper-origin X Y` | from jobs.yaml | Bottom-left corner offset from endstops (mm) |
| `--margin` | from jobs.yaml | Margin from paper edge (mm) |
| **Operational** | | |
| `--dry-run` | -- | Compute stats, no G-code |
| `--save-preview` | -- | Save vectorized preview image |
| `--save-vectors` | -- | Save pen_vectors.yaml |
| `--skip-corners` | -- | Skip alignment circles |

### Execution phases

| Phase | What happens |
|-------|-------------|
| 0 | Move to first corner, lower pen.  **Pause** -- operator inserts pen and checks depth. |
| 1 | Draw alignment circles at 4 corners (skipped with `--skip-corners`). Pause for paper alignment. |
| 2 | Draw border rectangle around the drawable area. |
| 3 | Draw all vectorized paths.  Progress % shown live. |
| 4 | Retract, return to centre, disable steppers. |

### Previews

When `--dry-run` is used, two previews are generated:
- `robot_control/images/previews/<name>_preview.png` -- machine-coordinate preview showing paper, margins, and paths
- `outputs/lineart_traces/<name>/preview.png` (with `--save-preview`) -- pixel-space vector preview

---

## Other scripts

### run_tracer.py -- Execute pre-generated pen vectors

Loads a `pen_vectors.yaml` and drives the robot through the drawing sequence.

```bash
.venv/bin/python robot_control/scripts/run_tracer.py \
  --traces-dir outputs/pen_traces_hard_bw \
  --image "peakpx (4) high res" --dry-run
```

### test_connection.py -- Verify Klipper API connectivity

```bash
.venv/bin/python robot_control/scripts/test_connection.py
```

### test_motors.py -- Hardware bring-up

Writes `printer.cfg`, restarts Klipper, tests motors, endstops, homing, and circles.

```bash
.venv/bin/python robot_control/scripts/test_motors.py
.venv/bin/python robot_control/scripts/test_motors.py --skip-to-homing --no-config-write
```

### test_motion.py -- Coordinated motion test

```bash
.venv/bin/python robot_control/scripts/test_motion.py
```

### calibrate.py -- Guided calibration

```bash
.venv/bin/python robot_control/scripts/calibrate.py               # All routines
.venv/bin/python robot_control/scripts/calibrate.py --endstops     # Endstop repeatability
.venv/bin/python robot_control/scripts/calibrate.py --steps-x      # X rotation_distance
.venv/bin/python robot_control/scripts/calibrate.py --z-heights    # Z seesaw heights
```

### run_job.py -- Execute calibration patterns

```bash
.venv/bin/python robot_control/scripts/run_job.py --pattern square
.venv/bin/python robot_control/scripts/run_job.py --pattern circle --interactive
```

### interactive_control.py -- Keyboard jog controller

```bash
.venv/bin/python robot_control/scripts/interactive_control.py
```

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Arrow keys | Jog X/Y | Page Up/Down | Jog Z |
| +/- | Change jog step | H | Home X Y |
| G | Go to position | P | Select pen |
| A | Select airbrush | U | Tool up |
| D | Tool down | O | Canvas origin |
| Esc | Emergency stop | Q | Quit |

---

## Running tests (no hardware needed)

```bash
.venv/bin/python -m pytest robot_control/tests/ -v
```

---

## Hardware overview

| Axis | Motors | Endstop | Home direction |
|------|--------|---------|----------------|
| X | 2 (dual gantry) | PG6 (STOP_0) | min (position 0) |
| Y | 1 | PG9 (STOP_1) | min |
| Z | 1 (seesaw) | PG10 (STOP_2) | max (position 80) |

Key parameters:
- **Motor**: 0.9deg (400 full steps/rev), DM542TE driver, 8 microsteps
- **Belt**: GT2 2mm pitch, 16T pulley -> `rotation_distance = 32 mm`
- **Workspace**: 450 x 320 x 80 mm (XYZ)
- **Max velocity**: 500 mm/s, max accel: 3000 mm/s^2

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: robot_control` | Not running from project root | `cd ~/Desktop/Airbrush-Painter` |
| `Klipper did not become ready` | MCU lost USB connection | Power cycle the Octopus board |
| Motors don't move | Enable pin polarity wrong | Check `enable_pin_inverted` in machine.yaml |
| Endstops always TRIGGERED | Polarity inverted | Toggle `endstop_polarity` in machine.yaml |
| Vibrations at high speed | Acceleration too high | Lower `max_accel_mm_s2` |
