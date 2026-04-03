# Robot Safety -- Critical Operating Procedures

**Read this entire document before powering the machine.**
Failure to follow these procedures can cause mechanical damage, fluid spills,
broken tools, or personal injury.

---

## 1. Homing -- Always Home Before Any Motion

The machine has **no absolute encoders**. After power-on or emergency stop, the
controller does not know where the axes are. Any motion command issued before
homing uses the last cached position, which is almost certainly wrong.

| Axis | Homing side | What happens if skipped |
|------|-------------|------------------------|
| X | min endstop | Gantry can crash into frame at either end |
| Y | min endstop | Gantry can crash into frame at either end |
| Z | **max** endstop | Tool can ram into bed (pen break, glass crack) |

**Rules:**

- **Always run `G28` (full home) before the first move after power-on or
  emergency stop.** The lineart tracer does this automatically; manual jog
  mode does not.
- **`G28 X Y` does NOT home Z.** The Job IR path (`HomeXY`) only homes X and
  Y. If Z has not been homed in the current session, its position is unknown.
  You must explicitly `G28 Z` or `G28` (all axes) before any Z move.
- **After `M18` (motors off), position is lost.** Re-home before resuming.
- **After an emergency stop (`M112` or API `emergency_stop`), re-home
  everything.** Klipper clears its position state on e-stop.

---

## 2. Z Axis -- The Most Dangerous Axis

Z homes to **max** (top). Work heights are near the **bottom** of travel.
A wrong Z value sends the tool into the bed at plunge speed.

| Z state | Current value | Purpose |
|---------|---------------|---------|
| `travel_mm` | 40.0 | Safe clearance for XY rapids |
| `pen_work_mm` | around 80.244 mm | Pen contact with paper surface |
| `airbrush_work_mm` | 0.0 | Airbrush spray height (+ `spray_height_mm`) |
| Soft limits | 0.0 -- 80.0 mm | Firmware rejects moves outside this range |

**Rules:**

- **Never change `z_contact_mm` or `pen_work_mm` without re-calibrating.**
  These values are determined by the Z height calibration routine (binary
  search in `calibration/routines.py`). A value even 0.5 mm too high crushes
  the pen tip or scratches the bed.
- **Z retract must happen before any XY rapid.** The G-code generator enforces
  this, but manual jog does not. Moving XY while the pen is down drags it
  across the paper and can bend or snap the tip.
- **Bed mesh must be loaded.** The bed is not perfectly flat. Without
  `BED_MESH_PROFILE LOAD=default`, Z compensation is disabled and the pen
  will press harder on high spots and lift off on low spots. Tolerance:
  ~0.15 mm planarity across the bed.
- **First probe point starts 2 mm above `z_contact_mm`.** If `z_contact_mm`
  is wrong, the first approach during bed mesh calibration can crash.

---

## 3. Y Axis Direction -- Coordinate Frame Mismatch

The machine uses **two different coordinate conventions** depending on the
code path:

| Path | Origin | +Y direction |
|------|--------|-------------|
| **Lineart tracer** (`run_lineart_tracer.py`) | Bottom-left (machine frame) | Up (away from operator) |
| **Job IR / G-code generator** | Top-left (image frame) | Down (toward operator) |

The lineart tracer applies a **180-degree pre-rotation** inside
`image_to_machine()` to compensate for the physical bed axis orientation.
The Job IR generator applies a Y-flip + offset via `canvas_to_machine()`.

**What breaks:**

- Mixing coordinates from one path into the other places the drawing
  **mirrored or rotated** relative to the paper. On a centered drawing this
  may look like a minor offset. On an off-center drawing it can place strokes
  **off the paper entirely**, into clamps, or into the frame.
- The software clamps XY to `[0, work_area]`, but clamped paths are
  **silently truncated**, not rejected. You get a partial drawing with no
  warning that geometry was clipped.

**Rules:**

- Use lineart tracer coordinates only with the lineart tracer.
- Use Job IR coordinates only with the Job IR generator.
- **Never hand-edit G-code** with raw coordinates unless you know which frame
  you are in. When in doubt, run a dry-run preview first.

---

## 4. Airbrush Needle -- Must Be Retracted Before Pumping Liquid

The airbrush needle controls whether liquid can flow through the nozzle.
It is actuated by a servo (pin PB6) through the Arduino controller.

| Klipper command | Servo state | Needle position | Liquid flow |
|----------------|-------------|-----------------|-------------|
| `SET_PIN PIN=servo_airbrush_needle VALUE=1` | ON (1200 us) | Retracted | **Enabled** |
| `SET_PIN PIN=servo_airbrush_needle VALUE=0` | OFF (1100 us) | Forward | **Blocked** |

**What breaks if you pump without retracting the needle:**

- Liquid has nowhere to go. Pressure builds inside the mixing manifold and
  tubing.
- The syringe pump stalls or skips steps trying to push against a closed
  nozzle.
- Liquid can back-flow through the refill valve or leak at tube fittings.
- If pressure builds enough, a tube can pop off a barb fitting, spraying ink
  onto the machine and electronics.

**Rules:**

- **Always retract the needle (`VALUE=1`) before dispensing liquid.**
- **Always close the needle (`VALUE=0`) before retracting the pump plunger**
  to avoid sucking air into the manifold.
- Wait for `needle_servo_delay_s` (0.5 s) after toggling the needle before
  commanding pump motion. The servo needs time to physically move.

---

## 5. Air Valve -- Sequence Matters

The air supply solenoid (pin PG15) controls compressed air to the airbrush.

**What breaks:**

- **Air ON + needle closed** = air pressurizes the nozzle body with no paint
  flow. When the needle is then retracted, a burst of pressurized air blows
  out whatever liquid was sitting in the nozzle, causing splatter.
- **Air OFF + needle open + pump dispensing** = liquid dribbles out under
  gravity/pump pressure with no atomization. It pools on the paper and can
  drip onto the bed or mechanics.
- **Air ON with no liquid in the manifold** = dry spray that deposits no
  paint but can blow dust and debris off the bed onto the paper.

**Correct sequence for spraying:**

1. Open air valve (`SET_PIN PIN=air_valve VALUE=1`)
2. Wait ~0.5 s for air pressure to stabilize
3. Retract needle (`SET_PIN PIN=servo_airbrush_needle VALUE=1`)
4. Begin pump dispense
5. ...spray...
6. Stop pump dispense
7. Close needle (`SET_PIN PIN=servo_airbrush_needle VALUE=0`)
8. Wait ~0.5 s
9. Close air valve (`SET_PIN PIN=air_valve VALUE=0`)

**Correct sequence for stopping:**

- Reverse of above. **Needle closes before air shuts off** to prevent the
  last puff of air from blowing residual liquid.

---

## 6. Syringe Pumps -- Homing, Travel Limits, and Refill

### 6.1 Homing

Each pump (pump_0 through pump_3) has a limit switch at the fully retracted
position. Homing drives the plunger in `homing_direction: 1` (retract) at
2.0 mm/s until the endstop triggers, then backs off 0.5 mm.

**What breaks if you skip pump homing:**

- The software does **not enforce travel limits** on un-homed pumps. The
  `check_travel_limit` function in `pump_controller.py` returns `True`
  (allow move) when the pump is not homed, even with `enforce_travel_limits:
  true`.
- Without a known position, the plunger can be driven past its physical
  travel (17.0 mm), jamming the leadscrew or cracking the syringe barrel.

### 6.2 Travel limits

| Parameter | Value |
|-----------|-------|
| Syringe volume | 1.0 ml |
| Plunger travel | 17.0 mm |
| Lead per step | 0.02 mm |
| Max dispense speed | 4.0 mm/s |
| Max retract speed | 2.0 mm/s |

- **Never command a move beyond 17.0 mm from home.** The software clamps
  this when homed, but manual G-code can bypass the check.
- **Backlash purge** (0.5 mm) is applied after homing to take up leadscrew
  slack. Skipping this means the first dispense has ~0.5 mm of dead travel.

### 6.3 Refill valve

The refill valve (servo on PB7) opens the fluid supply path to the syringe.

**What breaks:**

- **Valve open + pump dispensing** = liquid is pushed back into the supply
  reservoir instead of toward the nozzle. No paint reaches the airbrush.
- **Valve open + pump idle** = gravity can siphon liquid into or out of the
  syringe depending on reservoir height, flooding the manifold.
- **Valve closed + pump retracting** = vacuum builds in the syringe. Air
  leaks in through fittings, creating bubbles that cause inconsistent flow
  during the next dispense.

**Correct refill sequence:**

1. Close needle (`VALUE=0`)
2. Close air valve (`VALUE=0`)
3. Open refill valve (`SET_PIN PIN=servo_pump_refill VALUE=1`)
4. Wait `refill_valve_delay_s` (1.0 s)
5. Retract pump plunger to home
6. Wait for plunger to stop
7. Close refill valve (`SET_PIN PIN=servo_pump_refill VALUE=0`)
8. Wait `refill_valve_delay_s` (1.0 s)
9. Purge backlash (dispense 0.5 mm)

---

## 7. Speeds and Acceleration -- Mechanical Limits

### 7.1 XY gantry

| Parameter | Limit | Risk if exceeded |
|-----------|-------|-----------------|
| Max velocity | 500 mm/s | Belt skip, lost steps, frame shock |
| Max acceleration | 3000 mm/s^2 | Same + stepper stall (DM542TE) |
| Pen draw speed | 150 mm/s | Ink skip if too fast; mechanical stress if higher |
| Pen travel speed | 400 mm/s | Approaching planner limit |
| Airbrush feed speed | 100 mm/s | Spray pattern degrades at higher speeds |
| Airbrush travel speed | 160 mm/s | Conservative for loaded gantry |

### 7.2 Direction reversal pause

The DM542TE stepper drivers require a **0.5 s dwell** (`G4 P500`) on sharp
direction reversals at high speed. The G-code generator inserts this
automatically. If you hand-write G-code with back-to-back rapids in opposite
directions, **the driver can stall and lose steps silently**. There is no
encoder to detect this -- the machine continues with an offset that
accumulates.

### 7.3 Plunge speed

| Tool | Plunge speed | Why it matters |
|------|-------------|----------------|
| Pen | 20 mm/s | Pen tip is fragile; fast plunge bends or snaps it |
| Airbrush | 10 mm/s | Nozzle clearance is only `spray_height_mm` (3 mm) |

Z retract can be faster (up to 300 mm/s in jobs config) because the tool is
moving away from the surface.

---

## 8. Idle Timeout and Motor Disable

- **`idle_timeout_s: 30`** in machine config. Klipper disables stepper
  drivers after 30 s of no motion commands.
- When drivers are disabled, the gantry can be **moved by hand** or **drift
  under its own weight** (Z axis especially, since it may not have a
  counterweight or brake).
- The lineart tracer sets `SET_IDLE_TIMEOUT TIMEOUT=3600` during jobs to
  prevent mid-job disable. It resets to 30 s and sends `M18` at job end.

**Rules:**

- If you pause a job for more than 30 s, position may be lost. Re-home
  before resuming.
- After `M18`, the machine is in an unknown state. Always re-home.
- During manual work (loading paper, cleaning nozzle), either keep sending
  keep-alive commands or accept that you need to re-home afterward.

---

## 9. Calibration -- What Must Be Done and When

| Calibration | When required | What goes wrong if skipped |
|-------------|--------------|--------------------------|
| **Z height** | After any tool change, bed adjustment, or paper thickness change | Pen crashes into bed or hovers above paper |
| **Bed mesh** | After any bed leveling adjustment | Uneven pen pressure, scratching on high spots |
| **Steps/mm** | After belt replacement or pulley change | Drawing scale wrong, paths extend past paper |
| **Tool XY offset** | After mounting a different tool or re-mounting the same tool | Multi-tool jobs misaligned; airbrush sprays off-target |
| **Endstop repeatability** | Periodically (monthly) or after mechanical changes | Homing drift; must be < 0.1 mm XY, < 0.05 mm Z |
| **Pump homing** | Every session, before first dispense | Travel limits not enforced; plunger can jam |
| **Servo positions** | After servo replacement or Arduino reflash | Needle doesn't fully open/close; valve doesn't seal |

---

## 10. Emergency Procedures

### Emergency stop

- **Software:** Klipper API `emergency_stop` or `M112` via terminal. The
  CLI exposes this through `connection.py` -> `emergency_stop()`.
- **Effect:** All steppers immediately de-energize. Heaters off. Motion
  stops. **Position is lost.** You must re-home after clearing the e-stop.
- **Clearing:** Call `restart()` on the Klipper client, then `G28`.

### Fluid leak

1. Hit emergency stop to halt all pumps and motion.
2. Close the air valve manually (disconnect air supply if solenoid fails).
3. Close the refill valve (power off Arduino if servo is stuck open).
4. Mop up liquid immediately -- ink stains and IPA is flammable.
5. Check all tube connections before restarting.

### Pen break

1. Raise Z manually (jog up) or emergency stop.
2. Remove broken pen fragments from the bed before they scratch the surface.
3. Replace pen, re-calibrate Z height.

---

## 11. Software Safety Gaps (Known Limitations)

These are areas where the software does **not** protect you:

1. **No interlock between pumps and needle/air.** The software tracks valve
   and needle state for the UI, but there is no automatic check that
   prevents dispensing with a closed needle. The operator must follow the
   correct sequence.

2. **Un-homed pump travel limits are not enforced.** If a pump is not homed,
   `check_travel_limit` allows any move distance. Always home pumps first.

3. **XY clamping is silent.** The lineart tracer clamps paths to the work
   area without error. A drawing that extends past the paper boundary is
   quietly truncated. Always check the preview image.

4. **No encoder feedback.** Lost steps (from stalls, belt slip, or
   collisions) are undetectable. The machine continues with a cumulative
   offset. If a drawing looks shifted, stop and re-home.

5. **`M18` does not trigger a homing requirement.** After motors are
   disabled, Klipper does not force a re-home. The operator must remember
   to re-home.

6. **Cancel only takes effect after the current stroke.** In interactive
   execution mode, a cancel request waits for the in-progress stroke to
   finish. The tool is then raised using pen travel Z regardless of which
   tool is active.

7. **File execution mode continues after host disconnect.** When running
   G-code from the SD card, the host computer can disconnect and motion
   continues unsupervised on the MCU.

---

## 12. Pre-Flight Checklist

Run through this before every job:

- [ ] Machine powered on, Klipper connected (`klippy_uds` socket active)
- [ ] Arduino servo controller powered and showing "Servo controller ready"
- [ ] `G28` -- full home (all three axes)
- [ ] Bed mesh loaded (`BED_MESH_PROFILE LOAD=default`)
- [ ] Paper secured to bed, not overlapping clamps or frame
- [ ] `z_contact_mm` matches current paper thickness (re-calibrate if changed)
- [ ] Correct tool mounted and `tool XY offset` calibrated
- [ ] For airbrush jobs: air supply connected and regulator set
- [ ] For airbrush jobs: all pumps homed, syringes filled, refill valves closed
- [ ] For airbrush jobs: needle closes and opens cleanly (test with digital
  output toggle)
- [ ] Preview image reviewed -- drawing fits within paper bounds with margins
- [ ] Idle timeout extended if job will take more than 30 s between moves
- [ ] Emergency stop accessible (keyboard or physical button)
