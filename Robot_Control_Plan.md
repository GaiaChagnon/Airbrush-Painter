# Revised Architecture Plan: Klipper API UDS, Dual-Mode Execution, Testing & Calibration Focus

## Executive Summary

This plan addresses the critical flaws identified in the previous version while maintaining the good architectural decisions (Job IR, named Z states, controller/executor split). The key changes:

1. **Klipper API UDS** instead of `/tmp/printer` virtual serial—structured JSON protocol, not brittle text parsing
2. **Two execution modes**: File-run (production, safe) and Interactive (testing, calibration)
3. **Stroke-boundary streaming** for interactive mode, not line-count buffer management
4. **Coordinate transforms in generator**, not G92 runtime offsets
5. **Object model subscriptions** for state queries, not M114 text parsing
6. **Extensive calibration and testing tools** as primary deliverable

The scope is pen-only with clean extension points for airbrush.

---

## 1. Addressing the Critique

### Issue 1: `/tmp/printer` is Not a Unix Socket

**Critique**: The plan confused `/tmp/printer` (virtual serial/PTY) with the Klipper API Unix Domain Socket.

**Resolution**: Use the actual Klipper API UDS created via `klippy -a /path/to/klippy.sock`. This provides:
- Structured JSON request/response protocol
- Object model subscriptions for state
- Proper `emergency_stop` endpoint
- No text parsing required

### Issue 2: Line-Count Buffer Management is Wrong for Klipper

**Critique**: Managing "30-50 commands in flight" is a Marlin/GRBL pattern that doesn't fit Klipper's architecture.

**Resolution**: 
- **File-run mode**: Generate complete G-code file, use `virtual_sdcard` to print. Klipper manages all buffering.
- **Interactive mode**: Stream one stroke/path at a time, wait for motion complete between strokes. Pause is responsive because there's at most one stroke in flight.

### Issue 3: Streaming-Only is Fragile for Airbrush

**Critique**: With ink/air/valves, a stalled stream can leave the system in dangerous states.

**Resolution**: 
- File-run is the default for production jobs
- Interactive mode is explicit and intended for testing/calibration
- Macros are fail-safe (valves default closed)
- Future airbrush work biases toward preplanned execution

### Issue 4: G92 for Canvas Offset is Error-Prone

**Critique**: G92 changes the machine's position state and can accumulate errors.

**Resolution**: Apply all coordinate transforms (canvas offset, tool offset) in the G-code generator. The machine always operates in absolute machine coordinates. No runtime G92.

### Issue 5: M114 Parsing is Brittle

**Critique**: Klipper docs explicitly discourage parsing terminal output.

**Resolution**: Use Klipper API's `objects/query` endpoint to get `toolhead` position. Returns structured JSON, not text to parse.

### Issue 6: Emergency Stop Queueing

**Critique**: M112 through normal G-code path gets queued.

**Resolution**: Use Klipper API's dedicated `emergency_stop` endpoint for immediate halt. This bypasses the G-code queue.

### Issue 7: Compute Isolation

**Critique**: Running klippy on DGX during heavy compute risks timing issues.

**User Clarification**: The DGX Spark will be dedicated to this task only during operation. No concurrent heavy compute. This is acceptable.

---

## 2. System Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DGX SPARK (Dedicated)                          │
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │   Your Python    │                                                       │
│  │   Application    │                                                       │
│  │                  │                                                       │
│  │  ┌────────────┐  │                                                       │
│  │  │ Interactive│  │     ┌─────────────────────────────────────────────┐   │
│  │  │ Controller │  │     │                                             │   │
│  │  └─────┬──────┘  │     │              klippy process                 │   │
│  │        │         │     │                                             │   │
│  │  ┌─────┴──────┐  │     │  ┌─────────────┐    ┌──────────────────┐   │   │
│  │  │    Job     │  │     │  │ G-code      │    │ Motion Planner   │   │   │
│  │  │  Executor  │──┼─────┼─►│ Interpreter │───►│ (lookahead)      │   │   │
│  │  └────────────┘  │     │  └─────────────┘    └────────┬─────────┘   │   │
│  │                  │     │                              │             │   │
│  │  ┌────────────┐  │     │  ┌─────────────┐             │             │   │
│  │  │ Calibration│  │     │  │ Object Model│◄────────────┘             │   │
│  │  │   Tools    │  │     │  │ (state)     │                           │   │
│  │  └────────────┘  │     │  └──────┬──────┘                           │   │
│  │                  │     │         │                                   │   │
│  └──────────────────┘     │  ┌──────┴──────┐                           │   │
│           │               │  │ API Server  │◄── /tmp/klippy_uds        │   │
│           │               │  │ (JSON/UDS)  │                           │   │
│           │               │  └─────────────┘                           │   │
│           │               │         │                                   │   │
│           │               └─────────┼───────────────────────────────────┘   │
│           │                         │                                       │
│           │    Unix Domain Socket   │   Binary Protocol                     │
│           └─────────────────────────┘         │                             │
│                                               │ USB-C                       │
└───────────────────────────────────────────────┼─────────────────────────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │ Octopus Pro  │
                                         │ V1.0.1 H723  │
                                         │ (MCU)        │
                                         └──────────────┘
```

---

## 3. Klipper API Protocol

### Connection

The Klipper API server listens on a Unix Domain Socket. Start klippy with:

```bash
klippy /path/to/printer.cfg -a /tmp/klippy_uds -l /tmp/klippy.log
```

Connect via standard Unix socket:

```python
import socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/klippy_uds")
```

### Message Format

All messages are JSON objects terminated by `0x03` (ETX byte):

```
{"id": 1, "method": "info", "params": {}}\x03
```

Response:

```
{"id": 1, "result": {"state": "ready", ...}}\x03
```

### Key Endpoints

| Method | Purpose | Example |
|--------|---------|---------|
| `info` | Get server state | `{"method": "info"}` |
| `objects/list` | List available objects | `{"method": "objects/list"}` |
| `objects/query` | Query object state | `{"method": "objects/query", "params": {"objects": {"toolhead": ["position"]}}}` |
| `objects/subscribe` | Subscribe to state changes | `{"method": "objects/subscribe", "params": {"objects": {"toolhead": null}}}` |
| `gcode/script` | Execute G-code | `{"method": "gcode/script", "params": {"script": "G28 X Y"}}` |
| `gcode/subscribe_output` | Subscribe to terminal output (human display only) | `{"method": "gcode/subscribe_output"}` |
| `gcode/restart` | Restart after error | `{"method": "gcode/restart"}` |
| `emergency_stop` | Immediate halt | `{"method": "emergency_stop"}` |

### Terminal Output (Important Caveat)

`gcode/script` does **not** return terminal output in its JSON response. If you want terminal output for human display/debugging, you must subscribe via `gcode/subscribe_output`. Do **not** make correctness depend on parsing terminal output; it is explicitly discouraged.

### Position Query (Correct Way)

Instead of parsing M114 text:

```python
request = {
    "id": 123,
    "method": "objects/query",
    "params": {
        "objects": {
            "toolhead": ["position", "homed_axes"],
            "motion_report": ["live_position"]
        }
    }
}
# Response includes structured data:
# {"result": {"status": {"toolhead": {"position": [10.0, 20.0, 5.0, 0.0], "homed_axes": "xy"}}}}
```

### Emergency Stop (Correct Way)

Direct API call, not queued G-code:

```python
request = {"id": 999, "method": "emergency_stop"}
# Immediately halts motion, no queue delay
```

---

## 4. Execution Modes

### Mode A: File-Run (Production)

For complete jobs where reliability matters.

```
Job IR Operations
       │
       ▼
┌──────────────┐
│ G-code       │
│ Generator    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ job.gcode    │  (written to disk)
│ file         │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ virtual_     │  (Klipper's built-in file printer)
│ sdcard       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Klipper      │  (handles all buffering internally)
│ Motion       │
└──────────────┘
```

**Advantages:**
- Klipper manages entire motion queue
- Host can disconnect without affecting print
- Pause/resume handled by Klipper's `pause_resume` module
- Progress via `virtual_sdcard` object subscription
- No streaming fragility

**Key Implementation Detail (Virtual SD Card Files):**

This mode assumes the Python application can write the generated `.gcode` file **directly into** the host directory configured in Klipper's `[virtual_sdcard] path`. The Klipper API server is not a general-purpose file manager (upload/list/delete); it prints files that exist in that directory.

If you later want remote uploads and a richer file management API, add **Moonraker** and use its file manager endpoints instead of reinventing a file layer.

**API Calls:**
1. `gcode/script` with `SDCARD_PRINT_FILE FILENAME=job.gcode`
2. Subscribe to/query `virtual_sdcard` for progress (e.g., `is_active`, `progress`, `file_position`, `file_size`)
3. Optionally subscribe/query `print_stats` for higher-level status (`state`, `print_duration`, etc.)
4. `gcode/script` with `PAUSE` / `RESUME` / `CANCEL_PRINT` as needed

### Mode B: Interactive (Testing/Calibration)

For manual control, calibration, and testing where responsiveness matters.

```
User Input (keyboard, GUI)
       │
       ▼
┌──────────────┐
│ Interactive  │
│ Controller   │
└──────┬───────┘
       │
       │  One stroke/command at a time
       ▼
┌──────────────┐
│ Klipper API  │
│ gcode/script │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Wait for     │  (query toolhead state)
│ motion done  │
└──────┬───────┘
       │
       ▼
    Next command
```

**Key Principle: Stroke-Boundary Streaming**

Instead of buffering N lines:
- Send one complete stroke (travel + draw + lift)
- Include an explicit motion-finished barrier inside the stroke script
- Then send next stroke

This means pause is always responsive—at most one stroke is in motion.

**Critical Practical Fix: Always End Each Stroke Group with `M400`**

In Klipper, `gcode/script` completion corresponds to the script being accepted/processed, not a guarantee that motion has physically finished unless the script includes a blocking barrier. Always append `M400` ("wait for current moves to finish") to every interactive stroke group so `send_gcode(..., wait=True)` corresponds to "stroke is done".

**Implementation:**

```
For each stroke in interactive queue:
    1. Generate G-code for this stroke only
    2. Send via gcode/script
    3. Ensure the stroke script ends with M400 (motion barrier)
    4. Check for pause/cancel at stroke boundary
    5. Proceed to next stroke
```

---

## 5. File Architecture

```
src/
├── hardware/
│   ├── __init__.py
│   ├── klipper_client.py        # Klipper API UDS client
│   ├── job_executor.py          # File-run and interactive modes
│   └── interactive.py           # Keyboard controls, jog, manual commands
│
├── job_ir/
│   ├── __init__.py
│   └── operations.py            # Job IR dataclasses
│
├── calibration/
│   ├── __init__.py
│   ├── patterns.py              # Test pattern generators
│   ├── routines.py              # Calibration sequences
│   └── measurement.py           # Measurement input and calculation
│
└── gcode/
    ├── __init__.py
    └── generator.py             # Job IR → G-code conversion

configs/
└── hardware/
    └── machine.yaml             # All hardware configuration

scripts/
├── calibrate.py                 # Main calibration entry point
├── test_connection.py           # Verify Klipper API works
├── test_motion.py               # Basic motion tests
└── interactive_control.py       # Manual jog and control
```

---

## 6. File 1: Klipper Client

### Purpose

Low-level Klipper API communication. Handles socket connection, JSON message framing, request/response matching, and subscriptions.

### Responsibilities

| Responsibility | Details |
|----------------|---------|
| Connection management | Connect to UDS, handle reconnection |
| Message framing | JSON + 0x03 terminator |
| Request/response matching | Track by message ID |
| Subscription handling | Process async state updates |
| Error detection | Parse error responses, raise exceptions |
| Emergency stop | Direct API call, no queue |

### Interface

**Lifecycle**
- `connect()` → Establish socket connection, verify `info` response
- `disconnect()` → Clean socket close
- `is_connected` → Connection status property

**Commands**
- `send_gcode(script, wait=False)` → Execute G-code via API; if wait=True, block until script completes
- `emergency_stop()` → Immediate halt via dedicated endpoint
- `restart()` → Recover from emergency stop or error state

**Queries**
- `get_position()` → Query toolhead position, return structured Position object
- `get_status()` → Query printer state (ready, printing, error, etc.)
- `is_homed(axes)` → Check if specified axes are homed
- `is_idle()` → Check if toolhead is idle (no pending moves)

**Subscriptions**
- `subscribe(objects, callback)` → Subscribe to object state changes
- `unsubscribe()` → Remove subscriptions

**File Operations**
- `list_gcode_files()` → List files in virtual_sdcard directory
- `start_print(filename)` → Begin printing a file
- `pause_print()` → Pause current print
- `resume_print()` → Resume paused print
- `cancel_print()` → Cancel current print
- `get_print_progress()` → Query virtual_sdcard progress

### Message ID Tracking

Each request gets a unique ID. Responses are matched by ID. This allows:
- Multiple requests in flight
- Async subscription updates interleaved with responses
- Proper timeout handling per request

### Error Handling

| Error Type | Detection | Action |
|------------|-----------|--------|
| Connection failed | Socket exception | Raise `ConnectionError` |
| Klipper not ready | `info` returns non-ready state | Raise `KlipperNotReady` |
| G-code error | Response contains error field | Raise `GCodeError` with message |
| Timeout | No response within deadline | Raise `TimeoutError` |
| Shutdown | Klipper reports shutdown state | Raise `KlipperShutdown` |

---

## 7. File 2: Job Executor

### Purpose

Orchestrates job execution in both modes. Converts Job IR to G-code, manages execution state, provides progress feedback.

### Two Distinct Execution Paths

**Path A: `run_file(job_ir, output_path)`**

1. Convert entire Job IR to G-code
2. Write to file in virtual_sdcard directory
3. Call `start_print(filename)`
4. Subscribe to `virtual_sdcard` for progress
5. Return control to caller (non-blocking)
6. Provide `pause()`, `resume()`, `cancel()` methods

**Path B: `run_interactive(job_ir)`**

1. Convert Job IR to list of stroke groups
2. For each stroke group:
   - Generate G-code for that group
   - Send via `send_gcode(script, wait=True)`
   - Check pause/cancel flag
3. Fully blocking, runs in foreground

### Stroke Grouping for Interactive Mode

A "stroke group" is an atomic unit of execution:

```
Stroke Group:
  1. Rapid to start position (tool up)
  2. Lower tool
  3. Draw polyline (one or more G1 moves)
  4. Raise tool
  5. M400 (wait for motion complete)

Between groups: check for pause/cancel
```

This ensures:
- Pause always happens with tool raised
- Cancel leaves machine in safe state
- User can single-step through strokes

**Mode Transitions Must Be Safe by Construction:**
- Pause: complete current stroke group → ensure `ToolUp` + `M400` has executed → enter paused state.
- Cancel: prefer stroke-boundary cancel; if immediate stop is required use `emergency_stop` → recovery procedure (`gcode/restart` or `firmware_restart` as needed) → re-home XY → move to a defined park position.

### Interface

**File Mode**
- `run_file(operations, output_dir)` → Generate file, start print, return immediately
- `get_file_progress()` → Query print progress (percent, time remaining)
- `pause_file()` → Pause file print
- `resume_file()` → Resume file print
- `cancel_file()` → Cancel file print

**Interactive Mode**
- `run_interactive(operations, step_mode=False)` → Execute stroke by stroke; if step_mode=True, wait for confirmation between strokes
- `pause_interactive()` → Set pause flag (takes effect at next stroke boundary)
- `resume_interactive()` → Clear pause flag
- `cancel_interactive()` → Set cancel flag, abort after current stroke

**Common**
- `get_state()` → Return current state (idle, running_file, running_interactive, paused, error)
- `set_progress_callback(fn)` → Register callback for progress updates

### G-code Generation

All coordinate transforms happen here, not at runtime:

```
Canvas Position (from Job IR)
       │
       ├── Add canvas offset (canvas origin → machine origin)
       │
       ├── Add tool offset (pen vs airbrush XY offset)
       │
       ├── Apply Y-axis flip if needed (image frame → machine frame)
       │
       ▼
Machine Position (in G-code)
```

No G92 commands. All positions are absolute machine coordinates.

**Soft-Limit Validation (Cheap Insurance):**

Before generating/sending any moves (interactive) or writing the file (file-run), validate that:
- No commanded XY falls outside the machine work area.
- No pen-down motion occurs outside the canvas area (optionally include a safety margin).

Reject the job (preferred) or clamp only if you can prove clamping won't create unsafe pen-down motion.

### Z State Mapping

The Job IR uses semantic operations (`ToolDown`, `ToolUp`). Generator maps to configured Z values:

| Operation | Pen Tool | Airbrush Tool |
|-----------|----------|---------------|
| `ToolUp` (safe) | Z = 10.0 | Z = 10.0 |
| `ToolDown` (work) | Z = 20.0 | Z = 0.0 |

These values come from config, not hardcoded.

---

## 8. File 3: Job IR Operations

### Purpose

Define all possible job operations as immutable dataclasses. This vocabulary is the contract between vector data and G-code generation.

### Design Principles

1. **Semantic, not mechanical**: `ToolDown` not `G1 Z20`
2. **Units are millimeters**: All dimensions in mm
3. **Canvas-relative coordinates**: Origin at canvas corner, not machine home
4. **Tool-agnostic where possible**: Same `DrawPolyline` for pen and airbrush
5. **Extensible**: Add airbrush operations without modifying existing ones

### Operation Definitions

**Setup Operations**

| Operation | Parameters | Semantics |
|-----------|------------|-----------|
| `HomeXY` | (none) | Home X and Y axes using limit switches |
| `SelectTool` | tool: str | Choose active tool ("pen" or "airbrush") |

**Motion Operations**

| Operation | Parameters | Semantics |
|-----------|------------|-----------|
| `RapidXY` | x, y: float | Fast travel move, tool must be up |
| `LinearMove` | x, y: float; feed: optional float | Single line segment at draw speed |
| `DrawPolyline` | points: list of (x, y); feed: optional float | Connected line segments |

**Tool Operations**

| Operation | Parameters | Semantics |
|-----------|------------|-----------|
| `ToolUp` | (none) | Raise tool to safe height |
| `ToolDown` | (none) | Lower tool to work height |

**Future: Airbrush Operations**

| Operation | Parameters | Semantics |
|-----------|------------|-----------|
| `SetInkMix` | c, m, y: float (0-1) | Set CMY ratio for spray |
| `SprayOn` | (none) | Begin spraying |
| `SprayOff` | (none) | Stop spraying |
| `Purge` | volume_ml: float | Flush with solvent |

**Future: Pump Operations**

| Operation | Parameters | Semantics |
|-----------|------------|-----------|
| `RefillPump` | pump: str | Execute refill sequence |
| `PrimeLine` | pump: str | Push ink to nozzle |

### Grouping into Strokes

For interactive execution, operations are grouped:

```python
# A stroke is a logical drawing unit
Stroke = list[Operation]  # Typically: RapidXY, ToolDown, DrawPolyline..., ToolUp

# A job is a sequence of strokes
Job = list[Stroke]
```

The executor processes stroke by stroke in interactive mode.

---

## 9. File 4: Interactive Controller

### Purpose

Provide keyboard-driven manual control for testing and calibration. This is the primary interface during machine setup.

### Responsibilities

| Responsibility | Details |
|----------------|---------|
| Keyboard input | Capture arrow keys, hotkeys |
| Jog control | Move axes by configurable increments |
| Position display | Show current XYZ continuously |
| Tool control | Select pen/airbrush, raise/lower |
| Homing | Trigger home sequence |
| Single stroke execution | Run one stroke, wait, prompt for next |
| Speed override | Adjust feed rate during operation |

### Keyboard Mapping

| Key | Action |
|-----|--------|
| Arrow keys | Jog X/Y by current increment |
| Page Up/Down | Jog Z up/down |
| `+` / `-` | Increase/decrease jog increment (0.1, 1.0, 10.0 mm) |
| `H` | Home X and Y |
| `P` | Select pen tool |
| `A` | Select airbrush tool |
| `U` | Tool up (raise) |
| `D` | Tool down (lower) |
| `Space` | Execute next stroke (in step mode) |
| `Esc` | Emergency stop |
| `Q` | Quit interactive mode |

### Display

Terminal-based status display (updates continuously):

```
╔══════════════════════════════════════════════════════════╗
║  INTERACTIVE CONTROL                                     ║
╠══════════════════════════════════════════════════════════╣
║  Position:  X: 105.250   Y: 148.500   Z: 10.000         ║
║  Tool:      PEN (up)                                     ║
║  Jog Step:  1.0 mm                                       ║
║  State:     Ready                                        ║
╠══════════════════════════════════════════════════════════╣
║  [←→↑↓] Jog XY   [PgUp/Dn] Jog Z   [+/-] Step size      ║
║  [H] Home        [P] Pen   [A] Airbrush                  ║
║  [U] Up          [D] Down  [Space] Next stroke           ║
║  [Esc] E-STOP    [Q] Quit                                ║
╚══════════════════════════════════════════════════════════╝
```

### Safety

- Jog commands check soft limits before sending
- Tool down requires homed state
- E-stop always available via Esc key
- All jog moves execute with `wait=True` (one move at a time)

---

## 10. File 5: Calibration Patterns

### Purpose

Generate test patterns for calibration and verification. Each pattern is a function that returns Job IR operations.

### Pattern Library

**Basic Geometry**

| Pattern | Purpose | Output |
|---------|---------|--------|
| `square(size_mm)` | Verify XY scaling | 4 linear moves forming square |
| `rectangle(w, h)` | Verify aspect ratio | 4 linear moves |
| `cross(size_mm)` | Find center, verify axes | 2 perpendicular lines |
| `grid(rows, cols, spacing)` | Overall accuracy | Grid of lines |
| `diagonal(size_mm)` | Verify orthogonality | Diagonal line corner to corner |
| `circle(diameter, segments)` | Verify smooth motion | Polyline approximating circle |

**Calibration-Specific**

| Pattern | Purpose | Output |
|---------|---------|--------|
| `ruler_x(length, tick_spacing)` | Measure X steps/mm | Line with tick marks |
| `ruler_y(length, tick_spacing)` | Measure Y steps/mm | Line with tick marks |
| `crosshair_grid(spacing)` | Tool offset calibration | Grid of crosshairs |
| `speed_test(lengths, speeds)` | Find max reliable speed | Lines at different feeds |
| `acceleration_test(distance)` | Tune acceleration | Short moves vs long moves |
| `backlash_test(distance, repeats)` | Detect backlash | Forward/reverse pattern |
| `z_touch_pattern(positions)` | Z height calibration | Dots at different Z |

**Pen-Specific**

| Pattern | Purpose | Output |
|---------|---------|--------|
| `line_weight_test(speeds)` | Pen pressure vs speed | Parallel lines |
| `corner_test(angles)` | Sharp corner quality | Zigzag patterns |
| `fine_detail_test(spacing)` | Minimum feature size | Closely spaced lines |

### Pattern Parameters

All patterns accept:
- `origin`: Starting position on canvas (default: centered)
- `tool`: Which tool to use (default: current)
- `feed`: Drawing speed (default: from config)

### Output Format

Patterns return `list[Operation]`, ready for execution:

```python
ops = patterns.square(size_mm=50.0, origin=(80, 120))
executor.run_interactive(ops)
```

---

## 11. File 6: Calibration Routines

### Purpose

Guide the user through calibration procedures with measurements and calculations. Each routine is interactive: execute pattern, prompt for measurement, calculate correction, update config.

### Routine Library

**Steps Per MM Calibration**

1. Home machine
2. Move to starting position
3. Draw ruler pattern (100mm commanded)
4. Prompt user to measure actual length
5. Calculate correction factor
6. Display new `rotation_distance` value
7. Optionally update config file

**Tool Offset Calibration**

1. Select airbrush tool
2. Draw crosshair at known position
3. Select pen tool
4. Draw crosshair at same commanded position
5. Prompt user to measure offset between crosshairs
6. Calculate XY offset values
7. Update tool offset in config

**Z Seesaw Calibration**

1. Move to test position
2. Lower Z incrementally
3. Prompt user: "Is pen touching paper?" (Y/N)
4. Binary search to find exact touch point
5. Record as `pen_work_z`
6. Repeat for airbrush
7. Record as `airbrush_work_z`
8. Calculate travel Z (midpoint)

**Speed Calibration**

1. Draw test patterns at increasing speeds
2. Prompt user to identify highest quality line
3. Record as max recommended speed
4. Optionally test acceleration similarly

**Endstop Verification**

1. Command home
2. Query position (should be 0, 0)
3. Jog away from home
4. Command home again
5. Query position (should be 0, 0 again)
6. Report pass/fail

### Routine Interface

Each routine is a function that:
- Takes a `KlipperClient` and `config` 
- Runs interactively (prints prompts, reads input)
- Returns calibration results as dict
- Optionally writes updated config

```python
results = routines.calibrate_steps_per_mm(client, config, axis="X")
# results = {"axis": "X", "commanded": 100.0, "measured": 99.5, "correction": 1.005, "new_rotation_distance": 39.8}
```

---

## 12. Configuration File

### Structure

```yaml
# machine.yaml - Hardware configuration for Octopus Pro Painter

# Klipper API connection
connection:
  socket_path: "/tmp/klippy_uds"
  timeout_s: 5.0
  reconnect_attempts: 3

# Physical work area
machine:
  work_area_mm:
    x: 250.0
    y: 350.0
    z: 40.0

# Canvas position within work area  
canvas:
  offset_x_mm: 25.0      # Distance from X home to canvas left edge
  offset_y_mm: 25.0      # Distance from Y home to canvas bottom edge
  width_mm: 210.0        # A4 width
  height_mm: 297.0       # A4 height

# Z-axis seesaw states (your belt mechanism)
z_states:
  travel_mm: 10.0        # Both tools clear
  pen_work_mm: 20.0      # Pen contacts paper
  airbrush_work_mm: 0.0  # Airbrush at spray height

# Tool definitions
tools:
  pen:
    xy_offset_mm: [0.0, 0.0]     # Calibrate with crosshair test
    feed_mm_min: 1500            # Drawing speed
    travel_feed_mm_min: 6000     # Rapid speed
    plunge_feed_mm_min: 300      # Z lowering speed

  airbrush:
    xy_offset_mm: [0.0, 0.0]     # Reference tool (offsets are 0)
    feed_mm_min: 3000
    travel_feed_mm_min: 6000
    spray_height_mm: 3.0         # Height above canvas when spraying

# Motion limits (conservative for pen)
motion:
  max_velocity_mm_s: 100.0
  max_accel_mm_s2: 1000.0
  junction_deviation_mm: 0.05

# Interactive mode settings  
interactive:
  jog_increments_mm: [0.1, 1.0, 10.0, 50.0]
  default_jog_increment_mm: 1.0
  position_poll_interval_ms: 100

# Virtual SD card path (for file-run mode)
file_execution:
  gcode_directory: "/home/klipper/gcode_files"
  
# Future: pump configuration (disabled for pen-only)
pumps:
  enabled: false
  
# Future: servo configuration (disabled for pen-only)  
servos:
  enabled: false
```

### Configuration Validation

On load, validate:
- Socket path exists (or warn if klippy not running)
- Canvas fits within work area
- Tool offsets are within reasonable bounds
- Z states are ordered correctly (travel between work positions)
- Feed rates don't exceed motion limits

---

## 13. Klipper printer.cfg Requirements

Your Klipper configuration must include:

### MCU and Steppers

```ini
[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32h723xx_XXXXX

[stepper_x]
step_pin: PF13
dir_pin: PF12
enable_pin: !PF14
microsteps: 16
rotation_distance: 40        # Calibrate this
endstop_pin: ^PG6
position_endstop: 0
position_max: 250
homing_speed: 50

[stepper_y]
step_pin: PG0
dir_pin: PG1
enable_pin: !PF15
microsteps: 16
rotation_distance: 40        # Calibrate this
endstop_pin: ^PG9
position_endstop: 0
position_max: 350
homing_speed: 50

[stepper_y1]                 # Second Y motor
step_pin: PF11
dir_pin: PG3
enable_pin: !PG5
microsteps: 16
rotation_distance: 40        # Must match stepper_y

[stepper_z]
step_pin: PG4
dir_pin: PC1
enable_pin: !PA0
microsteps: 16
rotation_distance: 8         # Calibrate for your belt/pulley
position_min: -5             # Allow negative for seesaw
position_max: 40
# No endstop - calibrate manually
```

### Z Reference (Session Repeatability)

Named Z states (travel / pen_work / airbrush_work) are good, but they require a repeatable Z reference per session. With no Z endstop, you must choose at least one:
- Add a Z endstop at a known reference (recommended if practical).
- Use a mechanical hard-stop + a controlled "home to hard stop" procedure.
- Use a touch-off / calibration dock you run each session.

Treat Z calibration as **required at power-up** and **required after any fault / emergency stop / collision**, otherwise "Z=20" is not a meaningful physical height.

### Dual-Y Squareness (If Your Mechanics Can Rack)

If your Y gantry can rack (two Y motors without a mechanically tying belt/rigid gantry), homing to a single Y endstop does not guarantee squareness. If racking is possible, plan for:
- A second Y endstop (one per side) and a squaring procedure.

### Required Modules

```ini
[virtual_sdcard]
path: /home/klipper/gcode_files

[pause_resume]

[gcode_macro CANCEL_PRINT]
gcode:
    TURN_OFF_HEATERS          # No-op for airbrush, but required
    G91                       # Relative
    G0 Z5 F1200              # Lift
    G90                       # Absolute
    G0 X0 Y0 F6000           # Return home
    M84                       # Disable motors
```

### Tool Macros

```ini
[gcode_macro TOOL_PEN]
description: Select pen tool
gcode:
    G0 Z10 F1200             # Travel height first
    # Pen work Z will be commanded directly

[gcode_macro TOOL_AIRBRUSH]  
description: Select airbrush tool
gcode:
    G0 Z10 F1200             # Travel height first
    # Airbrush work Z will be commanded directly
```

### API Server

Start klippy with API socket:

```bash
~/klippy-env/bin/python ~/klipper/klippy/klippy.py \
    ~/printer.cfg \
    -a /tmp/klippy_uds \
    -l /tmp/klippy.log
```

---

## 14. Testing Strategy

### Level 1: Connection Tests

| Test | Procedure | Pass Criteria |
|------|-----------|---------------|
| Socket exists | Check `/tmp/klippy_uds` | File exists, is socket type |
| Connect | Open socket connection | No exception |
| Info query | Send `info` request | Response contains "ready" or "startup" |
| State query | `objects/query` on `webhooks.state` (or equivalent status object) | Returns "ready"/"startup"/"shutdown"/"error" |
| Position query | `objects/query` toolhead position / homed axes | Returns valid coordinates + homing state |
| Optional: terminal output | Subscribe via `gcode/subscribe_output` | Human-visible output stream (no parsing for correctness) |

### Level 2: Motion Tests

| Test | Procedure | Pass Criteria |
|------|-----------|---------------|
| Home X | Send `G28 X` | `homed_axes` includes "x" |
| Home Y | Send `G28 Y` | `homed_axes` includes "y" |
| Move X | Command X+50, query position | Position within 0.1mm |
| Move Y | Command Y+50, query position | Position within 0.1mm |
| Move Z | Command Z to each state | Position within 0.1mm |
| Rapid | G0 at max speed | Motion completes, no errors |
| Soft limits | Command beyond max | Klipper rejects, no motion |

### Level 3: Tool Tests

| Test | Procedure | Pass Criteria |
|------|-----------|---------------|
| Pen down | Command pen_work_z | Pen visibly touches paper |
| Pen up | Command travel_z | Pen clears paper |
| Airbrush down | Command airbrush_work_z | Airbrush at correct height |
| Tool switch | Alternate pen/airbrush | Both reach correct positions |

### Level 4: Drawing Tests

| Test | Procedure | Pass Criteria |
|------|-----------|---------------|
| Square | Draw 50mm square | Measures 50mm ± 0.5mm |
| Circle | Draw 50mm circle | Visually round, closed |
| Fine lines | Draw 0.5mm spaced lines | Lines distinct, no merging |
| Long job | Draw 1000-line pattern | Completes without error |

### Level 5: Control Tests

| Test | Procedure | Pass Criteria |
|------|-----------|---------------|
| Pause file | Start print, pause mid-job | Motion stops at stroke boundary |
| Resume file | Resume paused print | Continues correctly |
| Cancel file | Cancel mid-job | Returns home, clears state |
| Emergency stop | Trigger emergency_stop | Immediate halt |
| Recovery | Restart after e-stop | Can home and run again |

---

## 15. Implementation Phases

### Phase 0: Klipper Installation (Day 1)

**Tasks:**
1. Install Klipper on DGX Spark following official guide
2. Build firmware for Octopus Pro H723 (`make menuconfig`)
3. Flash via DFU mode
4. Create minimal `printer.cfg` with steppers only
5. Start klippy with `-a /tmp/klippy_uds`
6. Verify socket exists

**Verification:**
```bash
ls -la /tmp/klippy_uds
# Should show socket file
```

### Phase 1: Klipper Client (Day 2)

**Tasks:**
1. Implement socket connection with 0x03 framing
2. Implement request/response with ID matching
3. Implement `info`, `objects/query`, `gcode/script` methods
4. Implement `emergency_stop`
5. Write `test_connection.py`

**Verification:**
```bash
python scripts/test_connection.py
# Should show: Connected, position, firmware info
```

### Phase 2: Basic Motion (Day 3)

**Tasks:**
1. Implement `home()` using `G28 X Y`
2. Implement `get_position()` via object query
3. Implement `is_idle()` check
4. Complete printer.cfg with endstops
5. Write `test_motion.py`

**Verification:**
- Home command triggers physical motion
- Position queries return accurate values
- Soft limits prevent over-travel

### Phase 3: Interactive Controls (Day 4)

**Tasks:**
1. Implement keyboard capture (curses or similar)
2. Implement jog commands with increment selection
3. Implement position display loop
4. Implement tool selection (pen/airbrush)
5. Implement Z state control (up/down)

**Verification:**
- Arrow keys move machine
- Position display updates in real-time
- E-stop via Esc works immediately

### Phase 4: Calibration Tools (Day 5-6)

**Tasks:**
1. Implement pattern generators (square, ruler, cross, grid)
2. Implement steps/mm calibration routine
3. Implement Z seesaw calibration routine
4. Implement tool offset calibration routine
5. Test each calibration end-to-end

**Verification:**
- Patterns draw correctly
- Calibration routines produce valid corrections
- Updated values improve accuracy

### Phase 5: Job IR and Executor (Day 7-8)

**Tasks:**
1. Define all Job IR operations as dataclasses
2. Implement G-code generator with coordinate transforms
3. Implement file-run mode using virtual_sdcard
4. Implement interactive mode with stroke-boundary streaming
5. Implement pause/resume/cancel for both modes

**Verification:**
- File print runs complete job from file
- Interactive mode allows stroke-by-stroke execution
- Pause is responsive (within one stroke)

### Phase 6: Integration (Day 9-10)

**Tasks:**
1. Create main entry scripts (calibrate.py, interactive_control.py)
2. Add comprehensive error handling
3. Add logging throughout
4. Write end-to-end test with pen tracer output
5. Document all scripts and config options

**Verification:**
- Full pipeline: image → vectors → job → drawing
- All error conditions handled gracefully
- Logs provide debugging info

---

## 16. File Summary

| File | Purpose | Est. Lines |
|------|---------|------------|
| `src/hardware/klipper_client.py` | API UDS communication | ~300 |
| `src/hardware/job_executor.py` | File-run and interactive execution | ~350 |
| `src/hardware/interactive.py` | Keyboard control, jog, display | ~250 |
| `src/job_ir/operations.py` | Operation dataclasses | ~100 |
| `src/calibration/patterns.py` | Test pattern generators | ~200 |
| `src/calibration/routines.py` | Calibration procedures | ~300 |
| `src/gcode/generator.py` | Job IR → G-code | ~200 |
| `configs/hardware/machine.yaml` | Configuration | ~80 |
| `scripts/calibrate.py` | Calibration entry point | ~100 |
| `scripts/interactive_control.py` | Manual control entry point | ~50 |
| `scripts/test_*.py` | Test scripts | ~200 |
| **Total** | | **~2130** |

---

## 17. Key Architectural Principles

1. **Klipper API, not virtual serial**: Structured JSON protocol, not text parsing
2. **Two modes, one Job IR**: File-run for production, interactive for testing—same operations vocabulary
3. **Stroke-boundary streaming**: Interactive mode waits between strokes, not "N lines in buffer"
4. **Transforms in generator**: All coordinate math happens during G-code generation, no runtime G92
5. **Named Z states**: Semantic names (travel, pen_work, airbrush_work) for your seesaw mechanism
6. **Emergency stop via API**: Direct endpoint, not queued G-code
7. **Calibration-first**: Extensive tools to get the machine right before production use
8. **Fail-safe defaults**: Tool up between operations, valves closed by default (future)

---

This plan provides a solid foundation for pen operation with clear testing and calibration capabilities. The interactive mode makes setup and debugging straightforward. The file-run mode provides reliability for production jobs. The architecture extends cleanly to airbrush when that time comes.