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
| **Auto-reconnect** | **Detect MCU reset / Octopus power-cycle and re-establish connection** |
| Message framing | JSON + 0x03 terminator |
| Request/response matching | Track by message ID |
| Subscription handling | Process async state updates |
| Error detection | Parse error responses, raise exceptions |
| Emergency stop | Direct API call, no queue |

### Auto-Reconnect on MCU Reset

> **CHANGED Jan 29, 2026**: New requirement. When the Octopus is power-cycled or
> reset, Klipper enters a shutdown/disconnect state. The client must detect this
> and automatically re-establish the connection without manual intervention.

**Behavior:**
1. On any socket error (`BrokenPipeError`, `ConnectionResetError`, empty recv):
   - Mark connection as lost.
   - Begin reconnection loop: attempt to reconnect up to `reconnect_attempts` times,
     sleeping `reconnect_interval_s` between attempts (both from `machine.yaml`).
   - After reconnecting, send `FIRMWARE_RESTART` to re-initialize the MCU.
   - Wait for Klipper to reach "ready" state (poll `info` endpoint).
   - Re-subscribe to any active subscriptions.
2. If reconnection fails after all attempts, raise `KlipperConnectionLost`.
3. Callers can check `is_connected` property; long-running loops (calibration,
   interactive) should handle `KlipperConnectionLost` gracefully (warn user,
   offer retry).

**Why this matters**: During development and testing, the Octopus frequently gets
power-cycled or hits firmware errors. Without auto-reconnect, every test script
must be restarted manually.

### Interface

**Lifecycle**
- `connect()` → Establish socket connection, verify `info` response
- `disconnect()` → Clean socket close
- `reconnect()` → Attempt to re-establish a lost connection (called automatically)
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
| Connection failed | Socket exception on initial connect | Raise `ConnectionError` |
| **Connection lost** | **BrokenPipe / empty recv during operation** | **Auto-reconnect loop (see above)** |
| **Reconnect exhausted** | **All retry attempts failed** | **Raise `KlipperConnectionLost`** |
| Klipper not ready | `info` returns non-ready state | Wait + retry, then raise `KlipperNotReady` |
| G-code error | Response contains error field | Raise `GCodeError` with message |
| Timeout | No response within deadline | Raise `TimeoutError` |
| Shutdown | Klipper reports shutdown state | Attempt `FIRMWARE_RESTART`, then reconnect |

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
#
# This is the single source of truth for machine dimensions, stepper
# hardware, and tool configuration. The code reads this file at startup.
# Change values here — do NOT hardcode them in Python.

# ─── Klipper API connection ──────────────────────────────────────────
connection:
  socket_path: "/tmp/klippy_uds"
  timeout_s: 5.0
  reconnect_attempts: 5          # Retries if Octopus was power-cycled
  reconnect_interval_s: 2.0      # Seconds between reconnect attempts
  auto_reconnect: true           # Re-establish connection on MCU reset

# ─── Physical work area (measure your machine) ──────────────────────
# These are the ACTUAL travel limits of each axis in mm.
# Must match position_max in printer.cfg.
machine:
  work_area_mm:
    x: 250.0                    # CHANGE to actual X travel
    y: 350.0                    # CHANGE to actual Y travel
    z: 40.0                     # CHANGE to actual Z travel

# ─── Canvas position within work area ────────────────────────────────
canvas:
  offset_x_mm: 25.0             # Distance from X home to canvas left edge
  offset_y_mm: 25.0             # Distance from Y home to canvas bottom edge
  width_mm: 210.0               # A4 width (or actual canvas)
  height_mm: 297.0              # A4 height (or actual canvas)

# ─── Z-axis seesaw states ────────────────────────────────────────────
z_states:
  travel_mm: 10.0               # Both tools clear of paper
  pen_work_mm: 20.0             # Pen contacts paper
  airbrush_work_mm: 0.0         # Airbrush at spray height

# ─── Tool definitions ────────────────────────────────────────────────
tools:
  pen:
    xy_offset_mm: [0.0, 0.0]    # Calibrate with crosshair test
    feed_mm_s: 25.0              # Drawing speed (mm/s)
    travel_feed_mm_s: 80.0       # Rapid speed (mm/s)
    plunge_feed_mm_s: 5.0        # Z lowering speed (mm/s)

  airbrush:
    xy_offset_mm: [0.0, 0.0]
    feed_mm_s: 50.0
    travel_feed_mm_s: 80.0
    plunge_feed_mm_s: 5.0
    spray_height_mm: 3.0         # Height above canvas when spraying

# ─── Stepper / driver hardware ───────────────────────────────────────
# Documents the physical motor+driver configuration so that all code
# (calibration, motion tests, printer.cfg generation) stays consistent.
#
# All XY axes share the same motor/driver/belt/pulley combination.
# Z may differ — update z_rotation_distance separately when wired.
steppers:
  motor_type: "0.9deg"           # 400 native full steps/rev
  driver: "DM542TE"
  driver_pulses_per_rev: 1600    # Set via DM542TE DIP switches (4× for 0.9° motor)
  wiring: "common_anode"         # Octopus drives +, - to GND
  enable_pin_inverted: false     # common_anode → enable is NOT inverted (no "!")
  step_pulse_duration_s: 0.000005  # 5 µs — required for DM542TE via LS08 buffer
  direction_reversal_pause_s: 0.5  # 500 ms pause between rapid direction reversals

  # Klipper stepper settings (derived from above)
  klipper_microsteps: 4          # 400 × 4 = 1600 = driver_pulses_per_rev
  xy_rotation_distance: 32.0     # GT2 belt (2mm pitch) × 16T pulley = 32mm/rev
  z_rotation_distance: null      # TBD — fill in when Z is wired

  # Belt & pulley specs (for reference / calibration calculations)
  belt_type: "GT2"
  belt_pitch_mm: 2.0
  pulley_teeth: 16
  pulley_bore_mm: 5.0

# ─── Axis-to-Motor mapping (Octopus Pro V1.0.1 H723) ────────────────
# This section is informational — pin assignments live in printer.cfg.
# It exists so the code knows which Octopus slots are used.
axes:
  y:
    octopus_slot: "Motor 0"
    motors: 1                    # Single motor
    endstop_pin: "DIAG0 (PG6)"
    endstop_type: "NO_to_GND"   # Normally open, closes to ground
    homing_side: "min"           # Endstop at position_min (0)
  x:
    octopus_slot: "Motor 2_1 + Motor 2_2"
    motors: 2                    # Dual motor (stepper_x + stepper_x1)
    endstop_pin: "DIAG1 (PG9)"
    endstop_type: "NO_to_GND"
    homing_side: "min"
  z:
    octopus_slot: "TBD"
    motors: 1
    endstop_pin: null            # No endstop yet
    homing_side: null

# ─── Motion limits ───────────────────────────────────────────────────
# Start conservative; increase after running calibration patterns.
motion:
  max_velocity_mm_s: 80.0        # ~2.5 RPS at rotation_distance=32
  max_accel_mm_s2: 3000.0        # Conservative start
  junction_deviation_mm: 0.05
  homing_speed_mm_s: 30.0

# ─── Interactive mode settings ───────────────────────────────────────
interactive:
  jog_increments_mm: [0.1, 1.0, 10.0, 50.0]
  default_jog_increment_mm: 1.0
  position_poll_interval_ms: 100

# ─── File execution (virtual_sdcard) ─────────────────────────────────
file_execution:
  gcode_directory: "/home/klipper/gcode_files"

# ─── Future hardware (disabled) ──────────────────────────────────────
pumps:
  enabled: false
servos:
  enabled: false
```

### Configuration Validation

On load, validate:
- Socket path exists (or warn if klippy not running)
- Canvas fits within work area
- Canvas + offsets do not exceed machine travel
- Tool offsets are within reasonable bounds
- Z states are ordered correctly (travel between work positions)
- `xy_rotation_distance` matches `belt_pitch_mm × pulley_teeth`
- `klipper_microsteps × 400` equals `driver_pulses_per_rev`
- Feed rates don't exceed motion limits

---

## 12A. Hardware Specifications & Test Findings

> **CHANGED Jan 29, 2026**: Initial hardware test findings from motor bring-up.
> **CHANGED Jan 29, 2026 (rev 2)**: Added axis-to-motor mapping, endstop config,
> corrected rotation_distance to 32mm using actual GT2 16T pulley specs, added
> dual-motor X axis (Motor 2_1 + 2_2).

### Axis-to-Motor Mapping (Octopus Pro V1.0.1 H723)

| Axis | Octopus Slot | Step Pin | Dir Pin | Enable Pin | Role |
|------|-------------|----------|---------|------------|------|
| **Y** | Motor 0 | PF13 | PF12 | PF14 | Single Y motor |
| **X** (motor 1) | Motor 2_1 | PF11 | PG3 | PG5 | Dual X — left side |
| **X** (motor 2) | Motor 2_2 | PG4 | PC1 | PA0 | Dual X — right side |
| **Z** | TBD | TBD | TBD | TBD | Seesaw / pen-lift mechanism |

**Important**: Motor 0 is Y (not X as previously assumed). X uses two motors
(Motor 2_1 and 2_2) that move in tandem. In Klipper, these are `[stepper_x]` and
`[stepper_x1]`.

### Endstops

| Axis | Pin | Octopus Label | Switch Type | Wiring |
|------|-----|---------------|-------------|--------|
| Y | PG6 | DIAG0 | NO (normally open) | Connects to GND when triggered |
| X | PG9 | DIAG1 | NO (normally open) | Connects to GND when triggered |

- **Single-side homing only** — endstops are on one end of each axis, not both.
- Klipper pin config: `^PG6` / `^PG9` — the `^` enables the internal pull-up;
  pin reads HIGH normally, LOW when the switch triggers.
- With dual X motors and a single X endstop, there is **no automatic squareness
  guarantee**. The gantry must be mechanically constrained or a manual squaring
  procedure must be run each session.

### Belt & Pulley System

All XY axes use the same belt/pulley:

| Component | Spec |
|-----------|------|
| Belt | GT2, 6 mm wide, toothed |
| Pulley | 16 teeth, 5.00 mm bore H7, clamping screws |
| Belt pitch | 2.000 mm (GT2 standard) |
| **Travel per motor revolution** | **2 mm × 16 teeth = 32.000 mm** |

Z axis belt/pulley: TBD (may differ — document when wired).

### Motor & Driver Specifications (All Axes)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Motor type | 0.9°/step (400 native steps/rev) | All motors are 0.9°, NOT 1.8° |
| Driver | DM542TE | External driver, common-anode wiring to Octopus |
| Driver microstepping | 1600 pulses/rev | Set via DM542TE DIP switches (= 4× for 400-step motor) |
| Wiring | Common-anode | `-` pins connect to GND; Octopus drives `+` pins HIGH to activate |
| Enable pin polarity | Non-inverted (no `!`) | `VALUE=1` enables driver; `!` prefix would BREAK it |
| Step pulse duration | 5 µs (`step_pulse_duration: 0.000005`) | DM542TE spec ≥2.5 µs; 5 µs needed for LS08 buffer path |

### Klipper Configuration Derivation

For the math to be correct, Klipper's internal step count per "rotation" must match
the driver's pulses-per-revolution:

```
Klipper sends: full_steps_per_rev × microsteps = pulses per "Klipper rotation"
Driver expects: 1600 pulses = 1 physical revolution
Physical travel: 32 mm per revolution (GT2 16T)

Match condition: 400 × microsteps = 1600  →  microsteps = 4

Therefore:
  microsteps: 4
  rotation_distance: 32    (actual mm per revolution)
  steps_per_mm = (400 × 4) / 32 = 50
```

**Why microsteps=4 (not 16)?** With an external driver like the DM542TE, each Klipper
"step" is one physical pulse to the driver. The driver itself handles sub-step
interpolation. Setting `microsteps=4` means Klipper sends exactly 1600 pulses per
revolution, matching the driver 1:1. Setting `microsteps=16` would send 6400 pulses
per revolution, making the motor spin 4× more than Klipper expects (or requiring
`rotation_distance=128` to compensate — less intuitive).

### Tested Performance Limits (Motor 0, Y axis)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max tested velocity | 2.5 RPS (80 mm/s with correct rotation_distance=32) | Smooth operation |
| Max tested accel | 8000 mm/s² | No missed steps |
| Direction reversal | 500 ms pause required | Without pause, stalls at >1 RPS |

**Note**: The motor test script (`test_manual_stepper.py`) currently uses
`microsteps=16, rotation_distance=160` which is a 160/128=1.25× overestimate —
the motor physically turns ~20% less than commanded. This should be corrected to
`microsteps=4, rotation_distance=32` now that we know the actual pulley.

### Key Discoveries (from Bring-Up)

1. **Enable polarity is non-inverted** for common-anode wiring. With `!PF14`, the
   driver never enables and the motor doesn't lock.

2. **Standard `stepper_x` + `kinematics: cartesian` motion commands (`G1`, `FORCE_MOVE`)
   failed** during initial testing, while `STEPPER_BUZZ` (direct GPIO toggle) and
   hardware PWM on the step pin worked. Root cause was a combination of enable polarity
   timing and insufficient pulse width.

3. **`manual_stepper` bypasses kinematics** and was the path that finally worked with
   Klipper's motion planner. Production config should use standard `[stepper_x/y]`
   with `kinematics: cartesian`, but `manual_stepper` is the validated fallback
   for single-motor diagnostics.

4. **`step_pulse_duration: 0.000005`** (5 µs) eliminated vibration and missed steps.
   The DM542TE spec says ≥2.5 µs minimum, but the signal path through the LS08/HCT08
   logic buffer benefits from wider pulses.

5. **500 ms pause between direction reversals** prevents missed steps at high speed.

---

## 13. Klipper printer.cfg Requirements

> **CHANGED Jan 29, 2026 (rev 2)**: Complete rewrite. Motor 0 is now Y axis.
> X axis is dual-motor (Motor 2_1 + 2_2). rotation_distance corrected to 32mm
> (GT2 16T pulley). Endstops on DIAG0 (Y) and DIAG1 (X). See Section 12A.

Your Klipper configuration must include:

### MCU and Steppers

```ini
[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32h723xx_130028001051313234353230-if00

[printer]
kinematics: cartesian
max_velocity: 200               # mm/s — conservative; increase after calibration
max_accel: 3000                 # mm/s² — conservative; increase after calibration

# ─── Y AXIS ── Motor 0 ───────────────────────────────────────────────
# Single motor, single endstop on DIAG0
# 0.9° stepper, DM542TE at 1600 pulses/rev, GT2 16T pulley
[stepper_y]
step_pin: PF13
dir_pin: PF12
enable_pin: PF14                # NOT inverted — common-anode wiring
microsteps: 4                   # 400 × 4 = 1600 = driver pulses/rev
rotation_distance: 32           # GT2 belt × 16 teeth = 2mm × 16 = 32mm/rev
step_pulse_duration: 0.000005   # 5 µs — required for DM542TE via LS08 buffer
endstop_pin: ^PG6               # DIAG0, NO switch to GND, pull-up enabled
position_endstop: 0
position_max: 350               # CHANGE THIS to actual Y travel in mm
homing_speed: 30
homing_positive_dir: false      # Endstop is at the min (0) end

# ─── X AXIS ── Motor 2_1 (primary) ───────────────────────────────────
# Dual motor X axis — this is the primary with endstop
[stepper_x]
step_pin: PF11
dir_pin: PG3
enable_pin: PG5                 # NOT inverted — common-anode wiring
microsteps: 4
rotation_distance: 32           # GT2 16T — same as Y
step_pulse_duration: 0.000005
endstop_pin: ^PG9               # DIAG1, NO switch to GND, pull-up enabled
position_endstop: 0
position_max: 250               # CHANGE THIS to actual X travel in mm
homing_speed: 30
homing_positive_dir: false      # Endstop is at the min (0) end

# ─── X AXIS ── Motor 2_2 (secondary, mirrors stepper_x) ──────────────
# No endstop — moves in tandem with stepper_x
[stepper_x1]
step_pin: PG4
dir_pin: PC1
enable_pin: PA0                 # NOT inverted — common-anode wiring
microsteps: 4
rotation_distance: 32           # Must match stepper_x exactly
step_pulse_duration: 0.000005

# ─── Z AXIS ── TBD ───────────────────────────────────────────────────
# Z motor port, pulley, and endstop are not yet wired.
# Uncomment and fill in when Z is connected.
# [stepper_z]
# step_pin: TBD
# dir_pin: TBD
# enable_pin: TBD
# microsteps: 4
# rotation_distance: TBD        # Depends on Z belt/pulley — may differ from XY
# step_pulse_duration: 0.000005
# position_min: -5              # Allow negative for seesaw
# position_max: 40
# # No endstop — calibrate manually or add one
```

### Endstop Wiring Detail

Both endstops are **NO (normally open)** mechanical switches:

```
Switch open (not triggered):  pin floats → internal pull-up reads HIGH
Switch closed (triggered):    pin connects to GND → reads LOW
```

Klipper interprets `^PG6` as: enable pull-up, trigger on LOW. This is correct
for NO switches wired to GND.

**Physical location**: Both endstops are on one side only (the min/home end).
There are no max-end endstops — Klipper's `position_max` soft limit is the
only protection against over-travel.

### Z Reference (Session Repeatability)

Named Z states (travel / pen_work / airbrush_work) are good, but they require a
repeatable Z reference per session. With no Z endstop, you must choose at least one:
- Add a Z endstop at a known reference (recommended if practical).
- Use a mechanical hard-stop + a controlled "home to hard stop" procedure.
- Use a touch-off / calibration dock you run each session.

Treat Z calibration as **required at power-up** and **required after any fault /
emergency stop / collision**, otherwise "Z=20" is not a meaningful physical height.

### Dual-X Squareness

With two X motors and a single X endstop, there is no automatic squareness
guarantee. The X gantry can rack (skew) if one motor misses steps or the
mechanics allow twist. Mitigation options:
- **Mechanical**: Ensure the gantry is rigid enough that racking is physically
  impossible (preferred).
- **Second endstop**: Add one per side of the X axis, use Klipper's
  `[homing_override]` to home each motor independently and square the gantry.
- **Manual**: Run a squaring check each session using a calibration pattern.

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

## 18. Planned Full Rewrite of `robot_control/`

> **CHANGED Jan 29, 2026 (rev 2)**: The entire `robot_control/` package will be
> rewritten from scratch. This section serves as a checklist of every requirement
> that the new code must satisfy, derived from hardware testing and this plan.
> The existing code should be treated as reference only — do not patch it.

### Why Rewrite (Not Patch)

The existing code was written before hardware testing revealed:
1. Motor 0 is Y (not X) — all axis assumptions are wrong.
2. X is dual-motor (Motor 2_1 + 2_2) — not accounted for.
3. `rotation_distance` was wrong (40 or 160 vs. correct 32).
4. `microsteps` was wrong (16 vs. correct 4).
5. Auto-reconnect was not implemented.
6. Machine dimensions should come from config YAML, not hardcoded.
7. Endstop configuration was placeholder.
8. Feed rates were in mm/min in some places, mm/s in others (inconsistent).

### Requirements for the New Code

#### A. `machine.yaml` (Config File)

Must contain all the fields documented in Section 12. The code must:
- Read `machine.yaml` at startup as the single source of truth.
- Validate all fields (Section 12 validation checklist).
- Never hardcode machine dimensions, rotation_distance, feed rates, pin
  assignments, or stepper parameters — always read from config.
- Provide sensible error messages if fields are missing or invalid.

#### B. `klipper_client.py` (Klipper API Client)

Must implement:
- UDS connection with JSON + 0x03 framing (existing protocol is correct).
- **Auto-reconnect** on MCU reset / power-cycle (Section 6):
  - Detect broken socket (BrokenPipe, empty recv, ConnectionReset).
  - Retry loop: `reconnect_attempts` × `reconnect_interval_s` from config.
  - After reconnecting: send `FIRMWARE_RESTART`, wait for "ready" state.
  - Re-subscribe to any active object subscriptions.
  - Raise `KlipperConnectionLost` if all retries fail.
- `emergency_stop()` via dedicated API endpoint (not queued G-code).
- `get_position()` via `objects/query` (not M114 parsing).
- `wait_for_idle()` that polls toolhead state.
- All timeouts configurable via `machine.yaml`.

#### C. Axis & Motor Configuration

The code must understand:

| Axis | Klipper Section(s) | Motors | Endstop |
|------|--------------------|--------|---------|
| Y | `[stepper_y]` | Motor 0 (PF13/PF12/PF14) | DIAG0 (^PG6) |
| X | `[stepper_x]` + `[stepper_x1]` | Motor 2_1 (PF11/PG3/PG5) + Motor 2_2 (PG4/PC1/PA0) | DIAG1 (^PG9) |
| Z | `[stepper_z]` (TBD) | TBD | TBD |

All motor parameters for all axes:
- `microsteps: 4` (400 full steps × 4 = 1600 = driver pulses/rev)
- `rotation_distance: 32` (GT2 2mm pitch × 16 teeth)
- `step_pulse_duration: 0.000005` (5 µs)
- `enable_pin` is NOT inverted (common-anode wiring)

#### D. Homing

- Home X and Y only (Z has no endstop yet).
- `G28 X Y` — Klipper handles the homing sequence.
- After homing, verify position via `objects/query`.
- With single-side endstops, homing always goes to position_min (0).

#### E. printer.cfg Generation (Optional but Recommended)

Consider generating `printer.cfg` from `machine.yaml` so there is one config
to maintain. If not generating, at minimum validate that `machine.yaml` and
`printer.cfg` are consistent at startup (e.g., check `rotation_distance` match).

#### F. Feed Rate Convention

Standardize on **mm/s** throughout the Python code. Convert to Klipper's
`F` parameter (mm/min) only at the G-code generation boundary:
`F_value = feed_mm_s * 60.0`.

#### G. Direction Reversal Pause

When generating G-code that reverses direction at high speed (e.g., rapid
back-and-forth patterns), insert a `G4 P500` (500 ms dwell) or equivalent
pause between direction changes. Read the value from
`steppers.direction_reversal_pause_s` in config.

#### H. Calibration Routines

- `rotation_distance` default must be `32.0` (from config, not hardcoded).
- Steps/mm calibration must output the correct formula:
  `new_rotation_distance = old × (commanded / measured)`.
- Squareness check pattern for dual-X gantry.

### Files to Create (New Structure)

```
robot_control/
├── __init__.py
├── configs/
│   ├── __init__.py
│   ├── loader.py              # Load + validate machine.yaml
│   └── machine.yaml           # The config file (Section 12 spec)
├── hardware/
│   ├── __init__.py
│   ├── klipper_client.py      # UDS client with auto-reconnect
│   ├── job_executor.py        # File-run + interactive modes
│   └── interactive.py         # Keyboard jog, manual control
├── job_ir/
│   ├── __init__.py
│   └── operations.py          # Job IR dataclasses
├── calibration/
│   ├── __init__.py
│   ├── patterns.py            # Test pattern generators
│   ├── routines.py            # Guided calibration sequences
│   └── measurement.py         # Measurement math
├── gcode/
│   ├── __init__.py
│   └── generator.py           # Job IR → G-code (transforms here)
├── scripts/
│   ├── test_connection.py     # Verify Klipper API works
│   ├── test_motion.py         # Basic XY motion + endstop tests
│   ├── calibrate.py           # Calibration entry point
│   ├── run_job.py             # Production job runner
│   └── interactive_control.py # Manual control entry point
└── tests/
    ├── __init__.py
    ├── test_klipper_client.py # Mock UDS + reconnect tests
    ├── test_gcode_generator.py
    ├── test_job_ir.py
    └── test_calibration.py
```

### Key Differences from Existing Code

| Aspect | Old Code | New Requirement |
|--------|----------|-----------------|
| X axis motor | Motor 0 (PF13/PF12/PF14) | Motor 2_1 + 2_2 (dual) |
| Y axis motor | Not defined correctly | Motor 0 (PF13/PF12/PF14) |
| microsteps | 16 | 4 |
| rotation_distance | 40.0 (hardcoded default) | 32.0 (from config) |
| Feed rate units | Mixed mm/min and mm/s | mm/s everywhere, convert at G-code boundary |
| Reconnect | Not implemented | Auto-reconnect with retry loop |
| Config source | Some from YAML, some hardcoded | Everything from `machine.yaml` |
| Endstops | Placeholder pins | DIAG0 (^PG6) for Y, DIAG1 (^PG9) for X |
| Dual motors | Not supported | stepper_x + stepper_x1 |
| Z axis | Hardcoded pins | TBD, commented out until wired |

---

## 19. Diagnostic Test Scripts (Reference)

> Documents the test scripts created during motor bring-up, for future reference
> when debugging hardware issues.

| Script | Purpose | Status |
|--------|---------|--------|
| `test_manual_stepper.py` | Speed ramp + rapid 360° reversals via `manual_stepper` | **Working** — primary diagnostic tool |
| `test_motor_pwm_spin.py` | Direct hardware PWM on step pin (bypasses motion planner) | **Working** — proves hardware path is functional |
| `test_motor_0.py` | Initial STEPPER_BUZZ test | Superseded by `test_manual_stepper.py` |
| `test_speed_sweep.py` | Speed sweep via G1 moves | Did not work with original config |
| `test_force_move.py` | FORCE_MOVE testing | Did not work with original config |

**Important**: `test_manual_stepper.py` currently uses `microsteps=16, rotation_distance=160`
which has a ~20% scaling error. The correct values are `microsteps=4, rotation_distance=32`
(see Section 12A). Update the test script when running it next.

The `test_manual_stepper.py` configuration is the **canonical working reference** for
motor bring-up procedure (enable check, motion check, speed ramp). When bringing up
new axes (X dual motors, Z), start from its test sequence.

---

## 20. Changelog

| Date | Section | Change |
|------|---------|--------|
| Jan 29, 2026 | 12A (new) | Added hardware test findings from motor bring-up |
| Jan 29, 2026 | 13 | Fixed enable_pin polarity: `!PF14` → `PF14` (common-anode) |
| Jan 29, 2026 | 13 | Added `step_pulse_duration: 0.000005` to all stepper sections |
| Jan 29, 2026 | 13 | Changed motor assumption from 1.8° to 0.9° (400 steps/rev) |
| Jan 29, 2026 | 13 | Changed `rotation_distance` from `40` to `160` for XY steppers |
| Jan 29, 2026 | 13 | Added actual MCU serial ID |
| Jan 29, 2026 | 12 (config) | Added `steppers:` section to machine.yaml spec |
| Jan 29, 2026 | 12 (config) | Updated motion limits to tested values |
| Jan 29, 2026 | 18 (new) | Added pending code changes checklist |
| Jan 29, 2026 | 19 (new) | Added diagnostic test script reference |
| Jan 29, 2026 | 12A (rev 2) | Corrected axis-to-motor mapping: Y=Motor 0, X=Motor 2_1+2_2 (dual) |
| Jan 29, 2026 | 12A (rev 2) | Fixed rotation_distance: 160 → 32 (GT2 16T pulley = 2mm × 16 = 32mm) |
| Jan 29, 2026 | 12A (rev 2) | Fixed microsteps: 16 → 4 (400 × 4 = 1600 = driver pulses/rev) |
| Jan 29, 2026 | 12A (rev 2) | Added belt/pulley specs (GT2 6mm, 16T, 5mm bore) |
| Jan 29, 2026 | 12A (rev 2) | Added endstop configuration (DIAG0=Y, DIAG1=X, NO to GND) |
| Jan 29, 2026 | 13 (rev 2) | Complete rewrite of printer.cfg with correct axis mapping |
| Jan 29, 2026 | 13 (rev 2) | Added dual-X motor config (stepper_x + stepper_x1) |
| Jan 29, 2026 | 13 (rev 2) | Z axis marked TBD (not yet wired) |
| Jan 29, 2026 | 12 (rev 2) | Complete rewrite of machine.yaml spec with stepper/axis/endstop sections |
| Jan 29, 2026 | 12 (rev 2) | Added auto-reconnect config fields |
| Jan 29, 2026 | 12 (rev 2) | Standardized feed rates to mm/s |
| Jan 29, 2026 | 6 (rev 2) | Added auto-reconnect requirement for Klipper client |
| Jan 29, 2026 | 6 (rev 2) | Added `reconnect()` to client interface |
| Jan 29, 2026 | 6 (rev 2) | Updated error handling table with reconnect behavior |
| Jan 29, 2026 | 18 (rev 2) | Replaced patch list with full rewrite plan for robot_control/ |
| Feb 12, 2026 | 18 (impl) | Full rewrite of robot_control/ implemented: 28 files, ~6,300 lines |
| Feb 12, 2026 | 12 (impl) | machine.yaml created with all Section 12 fields (steppers, axes, mm/s feeds, reconnect) |
| Feb 12, 2026 | 6 (impl) | klipper_client.py with auto-reconnect, KlipperConnectionLost, FIRMWARE_RESTART, re-subscribe |
| Feb 12, 2026 | 18.F (impl) | Feed rates standardised to mm/s; conversion to mm/min at G-code boundary only |
| Feb 12, 2026 | 18.G (impl) | Direction reversal pause (G4 dwell) from steppers.direction_reversal_pause_s |

---

This plan provides a solid foundation for pen operation with clear testing and calibration capabilities. The interactive mode makes setup and debugging straightforward. The file-run mode provides reliability for production jobs. The architecture extends cleanly to airbrush when that time comes.