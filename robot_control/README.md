# robot_control

Motor control, calibration, and job execution for the Klipper-based pen/airbrush gantry.

Communicates with Klipper via its Unix Domain Socket (UDS) API using JSON + `0x03` (ETX) framing.  All hardware parameters live in `configs/machine.yaml` -- nothing is hardcoded in Python.

---

## Package structure

```
robot_control/
├── configs/                  Configuration layer
│   ├── machine.yaml          Single source of truth for all hardware params
│   ├── loader.py             YAML -> typed frozen dataclasses (MachineConfig)
│   └── printer_cfg.py        MachineConfig -> Klipper printer.cfg text
│
├── hardware/                 Klipper communication layer
│   ├── klipper_client.py     UDS client: connect, send G-code, query state
│   ├── job_executor.py       Runs Job IR via client (file or interactive mode)
│   └── interactive.py        Curses TUI for keyboard jog / manual control
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
├── scripts/                  CLI entry points (thin wrappers, no logic)
│   ├── test_motors.py        Hardware bring-up (motors, endstops, homing, circles)
│   ├── test_connection.py    Verify Klipper API reachable
│   ├── test_motion.py        Homing + coordinated XY move checks
│   ├── calibrate.py          Guided calibration entry point
│   ├── run_job.py            Run patterns or job files
│   ├── run_tracer.py         Trace pen vectors on the robot (A4 paper)
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

## File-by-file reference

### configs/machine.yaml

Single source of truth.  Every hardware value -- pin assignments, motor parameters, workspace dimensions, motion limits, tool feed rates -- is defined here.  Python code reads this at startup via `loader.py`.

Key sections:
- `connection` -- UDS socket path, MCU serial, reconnect settings
- `steppers` -- motor type, microsteps, rotation_distance, step_pulse_duration
- `axes` -- per-axis pin assignments (step/dir/enable), endstop pin + polarity, homing side
- `motion` -- max_velocity, max_accel, square_corner_velocity, homing_speed
- `machine.work_area_mm` -- travel limits (200x200 mm)
- `canvas` -- offset and size of the painting surface within the work area
- `tools` -- pen and airbrush feed rates and XY offsets

### configs/loader.py

Loads `machine.yaml` into typed, frozen dataclasses.  The root object is `MachineConfig`.

Key types:
- `MachineConfig` -- top-level; has helpers `get_tool()`, `canvas_to_machine()`, `get_z_for_tool()`
- `SteppersConfig` -- motor/driver/belt spec including `full_steps_per_rotation`
- `AxisConfig` -- per-axis pins (`PinConfig` tuple), endstop_pin, endstop_polarity, homing_side
- `MotionConfig` -- velocity, acceleration, square_corner_velocity, homing_speed
- `ConnectionConfig` -- socket path, mcu_serial, timeout, reconnect params

Validation (`_validate_config`) checks:
- Canvas fits inside work area
- `microsteps * full_steps_per_rotation == driver_pulses_per_rev`
- `rotation_distance == belt_pitch * pulley_teeth`
- Tool feeds don't exceed max velocity
- Pin count matches motor count per axis

### configs/printer_cfg.py

Generates a complete Klipper `printer.cfg` from `MachineConfig`.

`generate_printer_cfg(config) -> str` produces:
- `[mcu]` with the serial path from config
- `[printer]` with `kinematics: cartesian` and motion limits
- `[stepper_x]` + `[stepper_x1]` for dual-X (Klipper auto-syncs both motors)
- `[stepper_y]` with `homing_positive_dir: True` (Y homes to max)
- `[stepper_z]` dummy (never physically homed)
- `[force_move]` for per-stepper testing
- `[gcode_arcs]` for G2/G3 arc commands

The `step_pulse_duration` is always written as fixed-point (`0.000005`), never scientific notation.

### hardware/klipper_client.py

Full-featured Klipper API client over Unix Domain Socket.

`KlipperClient` provides:
- `connect()` / `disconnect()` / `reconnect()` -- connection lifecycle with auto-reconnect
- `send_gcode(script)` -- send G-code commands
- `get_position()` -> `Position(x, y, z, e)` -- current toolhead position
- `get_status()` -> `PrinterStatus(state, homed_axes, ...)` -- printer state
- `is_homed(axes)` / `is_idle()` / `wait_for_idle()` -- state queries
- `emergency_stop()` -- immediate halt
- `start_print()` / `pause_print()` / `resume_print()` / `cancel_print()` -- virtual SD card
- `subscribe()` / `unsubscribe()` -- object model subscriptions

Internally uses unique incrementing message IDs to match responses, filtering out async Klipper notifications.

### hardware/job_executor.py

Executes jobs (lists of Job IR operations) through the Klipper client.

`JobExecutor` modes:
- `run_file(ops, filename)` -- generates G-code, writes to virtual SD card, runs via Klipper's file executor.  Use `get_file_progress()` to monitor.
- `run_interactive(ops, step_mode)` -- sends strokes one at a time.  `step_mode=True` pauses between each stroke.
- `pause_interactive()` / `resume_interactive()` / `cancel_interactive()` -- mid-job control
- `set_progress_callback(fn)` -- called after each stroke with `ExecutorProgress`

States: `IDLE`, `RUNNING_FILE`, `RUNNING_INTERACTIVE`, `PAUSED`, `ERROR`.

### hardware/interactive.py

Curses-based terminal UI for manual machine control.

`InteractiveController.run()` blocks until quit.  Supports:
- Arrow keys for XY jog, Page Up/Down for Z jog
- +/- to change jog step size (0.1, 1.0, 10.0, 50.0 mm)
- H to home XY, P/A to select pen/airbrush, U/D for tool up/down
- O to go to canvas origin, Esc for emergency stop, Q to quit

Enforces soft limits and requires homing before tool-down.

### job_ir/operations.py

The vocabulary between vector data and G-code.  Every job action is an immutable, slotted dataclass using canvas-relative mm coordinates.

Operations:
- `HomeXY` -- home X and Y axes
- `SelectTool(tool)` -- choose "pen" or "airbrush"
- `RapidXY(x, y)` -- fast travel (tool must be up)
- `LinearMove(x, y, feed)` -- draw line at feed rate
- `DrawPolyline(points, feed)` -- connected line segments
- `DrawArc(x, y, i, j, clockwise, feed)` -- circular arc (G2/G3)
- `ToolUp` / `ToolDown` -- raise/lower to travel/work Z height
- `SetInkMix(c, m, y)`, `SprayOn`, `SprayOff`, `Purge` -- airbrush (future)

Helpers:
- `create_stroke(points, feed)` -> `[RapidXY, ToolDown, DrawPolyline, ToolUp]`
- `operations_to_strokes(ops)` -- splits a flat list into stroke groups at `ToolUp` boundaries

Types: `Stroke = list[Operation]`, `Job = list[Stroke]`.

### gcode/generator.py

Converts Job IR operations to G-code strings.

`GCodeGenerator(config)` provides:
- `generate(operations)` -> complete G-code program (header + ops + footer)
- `generate_stroke(stroke)` -> G-code for one stroke, ending with `M400`

Coordinate transforms applied here (not at runtime):
- Canvas (top-left origin, +Y down) -> machine (bottom-left origin, +Y up)
- Tool XY offset applied per active tool
- Feed rates converted from mm/s to G-code F parameter (mm/min)

Supports: G0 (rapid), G1 (linear), G2/G3 (arc), G28 (home).  Inserts direction-reversal dwells (G4) when rapids change direction >90 degrees.  Rejects positions outside soft limits with `GCodeError`.

### calibration/patterns.py

Pattern generators that return `list[Operation]` in canvas mm coordinates.

Geometry patterns: `square()`, `rectangle()`, `cross()`, `grid()`, `diagonal()`, `circle()`

Calibration patterns:
- `ruler_x()` / `ruler_y()` -- ruler with tick marks for measuring steps/mm
- `crosshair_grid()` -- grid of crosshairs for tool offset measurement
- `speed_test(speeds)` -- lines at increasing speeds
- `acceleration_test()` -- tests accel/decel behavior
- `backlash_test(repeats)` -- forward/reverse lines for backlash measurement
- `z_touch_pattern()` -- 5-point Z touch test (centre + 4 corners)

Pen-specific: `line_weight_test()`, `corner_test()`, `fine_detail_test()`

Composite: `calibration_suite()` -- runs all calibration patterns in sequence.

### calibration/routines.py

Guided interactive calibration: run a pattern, prompt for physical measurements, compute corrections.

Routines:
- `calibrate_steps_per_mm(client, config, axis)` -- draw ruler, measure, compute new `rotation_distance`
- `calibrate_z_heights(client, config)` -- binary search to find pen and airbrush work Z
- `calibrate_tool_offset(client, config)` -- crosshair pattern, measure offset between pen and airbrush
- `calibrate_speed(client, config)` -- draw lines at increasing speeds, user picks best
- `verify_endstops(client, config)` -- home twice, check position repeatability (delta < 0.1 mm)

All return `dict[str, Any]` with results.

### calibration/measurement.py

User input helpers and math for calibration routines.

- `get_float_input(prompt)` / `get_yes_no(prompt)` / `get_choice(prompt, choices)` -- terminal prompts
- `calculate_steps_correction(commanded, measured)` -> `MeasurementResult` (error, correction factor)
- `calculate_new_rotation_distance(old, commanded, measured)` -- corrected rotation_distance
- `calculate_tool_offset(pen_x, pen_y, ab_x, ab_y)` -- XY offset from crosshair centres
- `binary_search_z(prompt_fn, z_min, z_max, tolerance)` -- iterative Z height finder
- `format_calibration_summary(results)` -- pretty-print a results dict

---

## Data flow

```
machine.yaml
    │
    ▼
loader.py ──► MachineConfig (frozen dataclasses)
    │                │
    │                ├──► printer_cfg.py ──► ~/printer.cfg (Klipper config)
    │                │
    │                ├──► GCodeGenerator ──► G-code strings
    │                │         ▲
    │                │         │
    │                │    Job IR operations (HomeXY, DrawArc, LinearMove, ...)
    │                │         ▲
    │                │         │
    │                │    calibration/patterns.py
    │                │
    │                └──► KlipperClient ──► Klipper UDS ──► MCU ──► Motors
    │
    ▼
Validation (cross-field checks, soft limits, pin counts)
```

---

## Hardware overview

| Axis | Motors | Klipper sections | Endstop | Home direction |
|------|--------|------------------|---------|----------------|
| X    | 2 (dual gantry) | `[stepper_x]` + `[stepper_x1]` | PG6 (STOP_0) | min (position 0) |
| Y    | 1 | `[stepper_y]` | PG9 (STOP_1) | max (position 200) |
| Z    | 1 (dummy) | `[stepper_z]` | PG10 (unused) | not homed |

Key parameters (from `machine.yaml`):
- **Motor**: 0.9deg (400 full steps/rev), DM542TE driver, 8 microsteps = 3200 pulses/rev
- **Belt**: GT2 2mm pitch, 16T pulley -> `rotation_distance = 32 mm`
- **step_pulse_duration**: 5 us -- mandatory for DM542TE via LS08 buffer
- **Workspace**: 480 x 380 x 80 mm (XYZ, validated during bring-up)
- **Max velocity**: 400 mm/s, max accel: 3000 mm/s^2
- **Endstop polarity**: `^!` (pull-up + inverted, NO switch to GND)

---

## Scripts reference

All scripts run from the project root.  Activate the venv first: `source .venv/bin/activate`

### test_connection.py -- Verify Klipper API connectivity

Safe, no movement.  Checks socket connection, status query, position query, G-code send (M115).

```bash
python robot_control/scripts/test_connection.py
python robot_control/scripts/test_connection.py --socket /path/to/klippy_uds
```

### test_motors.py -- Hardware bring-up

The primary validation script.  Writes `printer.cfg`, restarts Klipper, then runs 6 phases:

| Phase | What it does | Can skip? |
|-------|-------------|-----------|
| 1 | Motor 0 (Y): FORCE_MOVE spin 3x back-and-forth | `--endstops-only` |
| 2 | Motor 1 (dummy Z): FORCE_MOVE spin | `--skip-motor1` |
| 3 | Motor 2_1+2_2 (X dual): G1 coordinated move | `--endstops-only` |
| 4 | Endstop verification: live polling while you press endstops | `--skip-to-homing` |
| 5 | Homing: G28 X Y | -- |
| 6 | Circle + speed ramp: G2 arcs at 20 to 190 mm/s | -- |

```bash
# Full sequence (writes printer.cfg, restarts Klipper)
python robot_control/scripts/test_motors.py

# Skip dummy Z motor
python robot_control/scripts/test_motors.py --skip-motor1

# Don't overwrite printer.cfg (assume it's already correct)
python robot_control/scripts/test_motors.py --no-config-write

# Jump straight to homing + circles
python robot_control/scripts/test_motors.py --skip-to-homing --no-config-write

# Custom speed and cycles
python robot_control/scripts/test_motors.py --speed 60 --cycles 5
```

### test_motion.py -- Coordinated motion test

Homes, then runs single-axis and combined XY moves, checking that reported positions match commanded positions.

Tests: X to 50, Y to 50, XY to 100/100, return to origin, rapid round-trip at max speed.

```bash
python robot_control/scripts/test_motion.py
python robot_control/scripts/test_motion.py --skip-home
```

### calibrate.py -- Guided calibration

Interactive routines that draw patterns, prompt for physical measurements, and compute corrections.

```bash
python robot_control/scripts/calibrate.py               # Run all routines
python robot_control/scripts/calibrate.py --endstops     # Endstop repeatability only
python robot_control/scripts/calibrate.py --steps-x      # X rotation_distance
python robot_control/scripts/calibrate.py --steps-y      # Y rotation_distance
python robot_control/scripts/calibrate.py --z-heights    # Z seesaw pen/airbrush heights
python robot_control/scripts/calibrate.py --tool-offset  # Pen vs airbrush XY offset
python robot_control/scripts/calibrate.py --speed        # Find best drawing speed
```

### run_job.py -- Execute patterns

Runs a named pattern from `calibration/patterns.py`.  Prepends homing automatically.

Available patterns: `square`, `rectangle`, `cross`, `grid`, `circle`, `diagonal`, `ruler-x`, `ruler-y`, `crosshair-grid`, `speed-test`, `backlash-test`, `z-touch`, `line-weight`, `corner-test`, `fine-detail`, `calibration-suite`, `acceleration-test`

```bash
python robot_control/scripts/run_job.py --pattern square
python robot_control/scripts/run_job.py --pattern circle --interactive
python robot_control/scripts/run_job.py --pattern grid --step          # pause between strokes
python robot_control/scripts/run_job.py --pattern speed-test --tool airbrush --feed 30
```

### interactive_control.py -- Keyboard jog controller

Curses TUI for manual machine control.  Auto-homes on startup.  If Klipper is in shutdown state (e.g. after MCU disconnect), the script automatically issues `FIRMWARE_RESTART` and waits for recovery.

```bash
.venv/bin/python robot_control/scripts/interactive_control.py
```

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Arrow keys | Jog X/Y | Page Up/Down | Jog Z |
| +/- | Change jog step | H | Home X Y |
| G | **Go to position** (type `x200 y150 z10`) | P | Select pen |
| A | Select airbrush | U | Tool up |
| D | Tool down | O | Canvas origin |
| Esc | Emergency stop | Q | Quit |

The **G** key opens an input line at the bottom of the TUI.  Type any combination of `x`, `y`, `z` coordinates (e.g. `x200`, `y150 z10`, `x100 y200`).  Supports formats: `x200`, `x=200`, `X200`.

### run_tracer.py -- Pen tracing on the robot

Loads a `pen_vectors.yaml` produced by the pen tracer pipeline, transforms image-frame coordinates to machine-frame coordinates, and drives the robot through the complete drawing sequence.

#### Pipeline overview

```
1. Image processing      run_pen_tracer_test.py
   (edge detection,      ──────────────────────►  pen_vectors.yaml
   hatching, path merge,                           pen_preview.png
   GNN+2-opt ordering)                             composite.png

2. Robot execution        run_tracer.py
   (load vectors,        ──────────────────────►  G-code via Klipper UDS
   coordinate transform,
   pen up/down, draw)
```

**Step 1 -- Generate pen vectors** (runs on any machine, no robot needed):

```bash
# Colour mode (CMY complement, gamut-aware hatching)
.venv/bin/python run_pen_tracer_test.py --mode color \
  --input "data/raw_images/hard/peakpx (4) high res.png" \
  --output outputs/pen_traces_hard

# Black & white mode (standalone pen drawing, full edge + dense hatching)
.venv/bin/python run_pen_tracer_test.py --mode bw \
  --input "data/raw_images/hard/peakpx (4) high res.png" \
  --output outputs/pen_traces_hard_bw

# Process an entire directory
.venv/bin/python run_pen_tracer_test.py --mode bw \
  --input data/raw_images/hard \
  --output outputs/pen_traces_hard_bw
```

Configs used: `configs/sim/pen_tracer_v2.yaml` (colour) or `configs/sim/pen_tracer_v2_bw.yaml` (B&W).

**Step 2 -- Run on the robot:**

```bash
# List available traced images
.venv/bin/python robot_control/scripts/run_tracer.py \
  --traces-dir outputs/pen_traces_hard_bw --list

# Dry run (stats only, no robot)
.venv/bin/python robot_control/scripts/run_tracer.py \
  --traces-dir outputs/pen_traces_hard_bw \
  --image "peakpx (4) high res" \
  --dry-run

# Full robot execution
.venv/bin/python robot_control/scripts/run_tracer.py \
  --traces-dir outputs/pen_traces_hard_bw \
  --image "peakpx (4) high res"

# Skip corner alignment (paper already placed)
.venv/bin/python robot_control/scripts/run_tracer.py \
  --traces-dir outputs/pen_traces_hard_bw \
  --image "peakpx (4) high res" \
  --skip-corners
```

#### Execution phases

| Phase | What happens |
|-------|-------------|
| 0 | Move to first corner, lower pen.  **Pause** -- operator inserts pen and checks depth. |
| 1 | Draw alignment circles at 4 corners (skipped with `--skip-corners`). Pause for paper alignment. |
| 2 | Draw border rectangle around the drawable area. |
| 3 | Draw all image paths (edges first, then hatching).  Progress % shown live. |
| 4 | Retract, return to centre, disable steppers. |

#### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image NAME` | -- | Subdirectory name in traces dir |
| `--vectors PATH` | -- | Direct path to `pen_vectors.yaml` (overrides `--image`) |
| `--traces-dir DIR` | `outputs/pen_traces_hard` | Where to look for traced images |
| `--list` | -- | List available images and exit |
| `--dry-run` | -- | Compute stats, no G-code |
| `--skip-corners` | -- | Skip alignment circles |
| **Z-axis** | | |
| `--z-contact` | 70.0 | Z position where pen touches paper (mm, higher = closer to bed) |
| `--z-retract` | 1.5 | Retract distance above contact (mm) |
| **Speeds** | | |
| `--draw-speed` | 30.0 | Drawing speed (mm/s) |
| `--travel-speed` | 100.0 | Travel speed (mm/s) |
| `--z-plunge-speed` | 20.0 | Pen down speed (mm/s) |
| `--z-retract-speed` | 50.0 | Pen up speed (mm/s) |
| `--accel` | 500.0 | XY acceleration (mm/s^2) |
| **Paper** | | |
| `--paper-origin X Y` | 25 25 | Offset from limit switches: X from X-min, Y from Y-max (mm) |
| `--margin` | 10.0 | Margin from paper edge (mm) |
| `--corner-radius` | 3.0 | Alignment circle radius (mm) |
| `--no-config-write` | -- | Don't regenerate printer.cfg |

#### Coordinate system

The paper is positioned using `--paper-origin X Y` which specifies offsets from the machine's limit switches:
- **X**: distance from the X endstop (X = 0)
- **Y**: distance from the Y endstop (Y = max)

With `--paper-origin 25 25`, the paper's top-left corner sits at machine position (25, 305) -- 25 mm from each endstop.  The image is scaled to fit the drawable area (A4 minus margins) with correct aspect ratio.

---

## Running tests (no hardware needed)

```bash
# All robot_control tests (108 tests)
.venv/bin/python -m pytest robot_control/tests/ -v

# Individual test files
.venv/bin/python -m pytest robot_control/tests/test_config_and_printer_cfg.py -v
.venv/bin/python -m pytest robot_control/tests/test_gcode_generator.py -v
.venv/bin/python -m pytest robot_control/tests/test_job_ir.py -v
.venv/bin/python -m pytest robot_control/tests/test_klipper_client.py -v
.venv/bin/python -m pytest robot_control/tests/test_calibration.py -v
```

The Klipper client tests use a mock UDS server -- no real hardware connection needed.

---

## Recommended test order with hardware

1. `test_connection.py` -- confirm Klipper is reachable
2. `test_motors.py` -- verify motors spin, endstops trigger, homing works
3. `test_motion.py` -- verify coordinated XY moves hit correct positions
4. `calibrate.py --endstops` -- confirm homing is repeatable
5. `calibrate.py --steps-x --steps-y` -- tune rotation_distance if measurements are off
6. `run_job.py --pattern square` -- first real pattern on paper

---

## Editing machine.yaml

After changing any value in `machine.yaml`, validate with:

```bash
.venv/bin/python -m pytest robot_control/tests/test_config_and_printer_cfg.py -v
```

To regenerate and deploy `printer.cfg` after config changes:

```bash
python robot_control/scripts/test_motors.py --skip-to-homing
```

This writes `~/printer.cfg`, restarts Klipper, homes, and runs the circle test.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: robot_control` | Not running from project root | `cd ~/Desktop/Airbrush-Painter` |
| `Klipper did not become ready` | MCU lost USB connection | Power cycle the Octopus board |
| `Lost communication with MCU` | USB disconnect or firmware crash | Script auto-attempts FIRMWARE_RESTART; if that fails, power cycle |
| Motors don't move | Enable pin polarity wrong | Check `enable_pin_inverted` in machine.yaml |
| Only 1 X motor moves | Missing `[stepper_x1]` in printer.cfg | Re-run `test_motors.py` (without `--no-config-write`) |
| Endstops always TRIGGERED | Polarity inverted | Toggle `endstop_polarity` between `^!` and `^` in machine.yaml |
| Vibrations at high speed | Acceleration too high | Lower `max_accel_mm_s2` (tested safe: 3000) |
| Audible speed changes in circles | G1 segments instead of G2 arcs | Use `DrawArc` operations or G2/G3 commands |
| Y homes to wrong end | `homing_side` wrong | Set `homing_side: "max"` for Y in machine.yaml |
