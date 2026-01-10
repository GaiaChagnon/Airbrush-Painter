# Robot Control

Isolated control system for the Klipper-based pen/airbrush robot. Communicates with Klipper via Unix Domain Socket API for motion control, calibration, and job execution.

## Overview

This package provides:

- **Klipper API Client**: Low-level UDS communication with JSON message framing
- **Job Executor**: File-run and interactive execution modes
- **Interactive Controller**: Curses-based terminal UI for manual control
- **Calibration Tools**: Test patterns and guided calibration routines
- **G-code Generator**: Job IR to G-code with coordinate transforms

## Requirements

- Linux (uses Unix Domain Sockets and curses)
- Python 3.10+
- Klipper running with API socket enabled
- PyYAML (`pip install pyyaml`)

## Directory Structure

```
robot_control/
├── hardware/           # Klipper communication and execution
│   ├── klipper_client.py   # UDS API client
│   ├── job_executor.py     # File-run and interactive modes
│   └── interactive.py      # Terminal UI controller
├── job_ir/             # Intermediate representation
│   └── operations.py       # Operation dataclasses
├── calibration/        # Calibration tools
│   ├── patterns.py         # Test pattern generators
│   ├── routines.py         # Guided calibration procedures
│   └── measurement.py      # Measurement helpers
├── gcode/              # G-code generation
│   └── generator.py        # Job IR -> G-code
├── configs/            # Configuration
│   ├── machine.yaml        # Hardware configuration
│   └── loader.py           # Config loading and validation
├── scripts/            # Entry points
│   ├── calibrate.py        # Calibration CLI
│   ├── interactive_control.py  # Manual control
│   ├── run_job.py          # Execute jobs
│   ├── test_connection.py  # Connection test
│   └── test_motion.py      # Motion test
└── tests/              # Unit tests
```

## Setup

### 1. Klipper Configuration

Ensure Klipper is started with the API socket enabled:

```bash
~/klippy-env/bin/python ~/klipper/klippy/klippy.py \
    ~/printer.cfg \
    -a /tmp/klippy_uds \
    -l /tmp/klippy.log
```

### 2. Required Klipper Modules

Add to your `printer.cfg`:

```ini
[virtual_sdcard]
path: /home/klipper/gcode_files

[pause_resume]

[gcode_macro TOOL_PEN]
description: Select pen tool
gcode:
    G0 Z10 F1200  ; Travel height first

[gcode_macro TOOL_AIRBRUSH]
description: Select airbrush tool
gcode:
    G0 Z10 F1200  ; Travel height first

[gcode_macro CANCEL_PRINT]
gcode:
    G91
    G0 Z5 F1200
    G90
    G0 X0 Y0 F6000
    M84
```

### 3. Machine Configuration

Edit `robot_control/configs/machine.yaml` to match your hardware:

```yaml
# Key settings to customize:
connection:
  socket_path: "/tmp/klippy_uds"

machine:
  work_area_mm:
    x: 250.0
    y: 350.0
    z: 40.0

canvas:
  offset_x_mm: 25.0    # Distance from home to canvas edge
  offset_y_mm: 25.0
  width_mm: 210.0      # A4 width
  height_mm: 297.0     # A4 height

z_states:
  travel_mm: 10.0      # Both tools clear
  pen_work_mm: 20.0    # Pen contacts paper
  airbrush_work_mm: 0.0

tools:
  pen:
    xy_offset_mm: [0.0, 0.0]
    feed_mm_min: 1500
```

## Usage

### Test Connection

Verify Klipper API is accessible:

```bash
python -m robot_control.scripts.test_connection
```

### Test Motion

Run motion verification tests:

```bash
python -m robot_control.scripts.test_motion
```

### Interactive Control

Launch terminal UI for manual control:

```bash
python -m robot_control.scripts.interactive_control
```

**Controls:**
- Arrow keys: Jog X/Y
- Page Up/Down: Jog Z
- +/-: Change jog increment
- H: Home X and Y
- P/A: Select pen/airbrush
- U/D: Tool up/down
- O: Go to canvas origin
- Esc: Emergency stop
- Q: Quit

### Calibration

Run calibration routines:

```bash
# Full calibration sequence
python -m robot_control.scripts.calibrate

# Individual calibrations
python -m robot_control.scripts.calibrate --steps-x
python -m robot_control.scripts.calibrate --steps-y
python -m robot_control.scripts.calibrate --z-heights
python -m robot_control.scripts.calibrate --tool-offset
python -m robot_control.scripts.calibrate --speed
python -m robot_control.scripts.calibrate --endstops
```

### Run Test Patterns

Draw test patterns for verification:

```bash
# Draw a square
python -m robot_control.scripts.run_job --pattern square

# Draw with interactive mode (pause between strokes)
python -m robot_control.scripts.run_job --pattern grid --interactive

# Draw calibration suite
python -m robot_control.scripts.run_job --pattern calibration-suite

# Dry run (generate G-code without executing)
python -m robot_control.scripts.run_job --pattern circle --dry-run
```

**Available patterns:**
- Basic: square, rectangle, cross, grid, circle, diagonal
- Calibration: ruler-x, ruler-y, crosshair-grid, speed-test, backlash-test
- Pen-specific: z-touch, line-weight, corner-test, fine-detail
- Suite: calibration-suite

## Execution Modes

### File-Run Mode (Production)

Generates complete G-code file and prints via Klipper's `virtual_sdcard`:

- Klipper manages entire motion queue
- Host can disconnect without affecting print
- Progress via virtual_sdcard object subscription
- Pause/resume handled by Klipper

### Interactive Mode (Testing/Calibration)

Stroke-by-stroke execution with M400 barriers:

- Pause takes effect at next stroke boundary
- Step mode allows single-stroke execution
- Tool always raised between strokes
- Responsive cancel/emergency stop

## API Usage

### Basic Job Execution

```python
from robot_control.configs.loader import load_config
from robot_control.hardware.klipper_client import KlipperClient
from robot_control.hardware.job_executor import JobExecutor
from robot_control.job_ir.operations import (
    HomeXY, ToolUp, ToolDown, RapidXY, DrawPolyline
)

# Load configuration
config = load_config()

# Create operations
ops = [
    HomeXY(),
    ToolUp(),
    RapidXY(x=50, y=50),
    ToolDown(),
    DrawPolyline(points=((50, 50), (100, 50), (100, 100), (50, 100), (50, 50))),
    ToolUp(),
]

# Execute
with KlipperClient(config.connection.socket_path) as client:
    executor = JobExecutor(client, config)
    executor.run_interactive(ops)
```

### Using Calibration Patterns

```python
from robot_control.calibration import patterns
from robot_control.hardware.job_executor import JobExecutor

# Generate a test pattern
ops = patterns.square(size_mm=50, origin=(80, 120))

# Execute with progress callback
def on_progress(progress):
    print(f"Progress: {progress.progress_percent:.1f}%")

executor.set_progress_callback(on_progress)
executor.run_interactive(ops)
```

## Running Tests

```bash
# From project root
pytest robot_control/tests/ -v

# Run specific test module
pytest robot_control/tests/test_job_ir.py -v
pytest robot_control/tests/test_gcode_generator.py -v
pytest robot_control/tests/test_calibration.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Entry Point Scripts                          │
│  calibrate.py  interactive_control.py  run_job.py               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                        Job Executor                              │
│  - File-run mode (virtual_sdcard)                               │
│  - Interactive mode (stroke-by-stroke)                          │
│  - Pause/resume/cancel state machine                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                       G-code Generator                           │
│  - Job IR → G-code conversion                                   │
│  - Canvas → machine coordinate transforms                       │
│  - Z state mapping (travel/pen_work/airbrush_work)             │
│  - Soft limit validation                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                       Klipper Client                             │
│  - Unix Domain Socket connection                                │
│  - JSON message framing (ETX terminated)                        │
│  - Request/response ID matching                                 │
│  - Object subscriptions for state                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                     /tmp/klippy_uds
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                        Klipper (klippy)                         │
│  - Motion planner with lookahead                                │
│  - Stepper control via MCU                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Klipper API, not virtual serial**: Structured JSON protocol, not text parsing
2. **Two modes, one Job IR**: File-run for production, interactive for testing
3. **Stroke-boundary streaming**: Interactive mode waits between strokes, not "N lines in buffer"
4. **Transforms in generator**: All coordinate math during G-code generation, no runtime G92
5. **Named Z states**: Semantic names for seesaw mechanism (travel, pen_work, airbrush_work)
6. **Emergency stop via API**: Direct endpoint, not queued G-code

## Troubleshooting

### Connection Failed

1. Check socket exists: `ls -la /tmp/klippy_uds`
2. Verify Klipper is running with `-a` flag
3. Check socket permissions

### Position Queries Fail

1. Ensure Klipper is in "ready" state
2. Home the machine first (`G28 X Y`)

### Motion Not Executing

1. Check Klipper logs: `tail -f /tmp/klippy.log`
2. Verify homing state: machine must be homed before moves
3. Check for soft limit violations

### Curses Display Issues

1. Ensure terminal supports ANSI/curses
2. Try resizing terminal window
3. Check `TERM` environment variable
