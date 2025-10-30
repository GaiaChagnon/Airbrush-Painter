# Two-Bounds Architecture Implementation

## Overview

Implemented separation of **machine physical bounds** and **canvas/painting bounds** to support:
- Canvas positioned anywhere within machine work area (can be offset from origin)
- Multiple paper sizes on the same machine  
- Explicit acceleration and speed limits in configuration

## Status: ✅ COMPLETE

**All tasks completed**:
1. ✅ Machine schema updated (`configs/schema/machine.v1.yaml`)
2. ✅ Machine config updated with example (`configs/machine_grbl_airbrush_v1.yaml`)
3. ✅ Validators updated (`src/utils/validators.py`)
4. ✅ `get_stroke_bounds()` returns canvas bounds
5. ✅ Coordinate transform updated to handle canvas offset
6. ✅ All `generate_*_gcode()` call sites updated (removed `work_area_mm` parameter)
7. ✅ `gcode_vm.py` updated to use `machine_cfg.acceleration` from config
8. ✅ Tests updated to use new bounds structure
9. ✅ `generate_pen_gcode()` coordinate transform updated
10. ✅ Both test suites pass (comprehensive + extended)

---

## Architecture Changes

### Before (Single Bounds)
```yaml
work_area_mm:       # Used for BOTH machine and canvas
  x: 210.0
  y: 297.0  
  z: 30.0
```

### After (Two Bounds)
```yaml
# Machine physical limits (origin at 0,0)
work_area_mm:
  x: 250.0          # Machine can travel 250mm
  y: 350.0          # Machine can travel 350mm
  z: 40.0

# Canvas bounds (painting surface, can be offset)
canvas_mm:
  x_min: 20.0       # Canvas starts 20mm from origin
  x_max: 230.0      # Canvas ends at 230mm (210mm wide A4)
  y_min: 26.5       # Centered vertically
  y_max: 323.5      # 297mm tall A4

# NEW: Explicit acceleration limits
acceleration:
  max_xy_mm_s2: 1000.0
  max_z_mm_s2: 500.0

# NEW: Rapid speed in feeds
feeds:
  max_xy_mm_s: 300.0
  max_z_mm_s: 20.0
  rapid_mm_s: 200.0    # NEW
```

---

## Key Concepts

### 1. Machine Coordinates (Absolute)
- Origin at (0, 0) - machine home position
- XY range: [0, work_area_mm.x] × [0, work_area_mm.y]
- Used for: Soft limits, G-code commands, safety checks

### 2. Canvas Coordinates (Relative to Canvas)
- Origin at canvas top-left in image frame
- XY range: [0, canvas_width] × [0, canvas_height]
- Used for: Stroke coordinates, painting, RL policy

### 3. Coordinate Transform
**Image Frame** (relative to canvas) → **Machine Frame** (absolute):
```python
# Old (assumed canvas at origin):
machine_x = image_x
machine_y = canvas_height - image_y  # Flip Y

# New (canvas can be offset):
machine_x = image_x + canvas.x_min
machine_y = (canvas_height - image_y) + canvas.y_min
```

---

## Schema Changes

### Machine Schema (`configs/schema/machine.v1.yaml`)

**Added Fields**:
- `canvas_mm` (required):
  - `x_min`: Canvas left edge (mm from machine origin)
  - `x_max`: Canvas right edge
  - `y_min`: Canvas bottom edge  
  - `y_max`: Canvas top edge
- `acceleration` (required):
  - `max_xy_mm_s2`: XY acceleration limit
  - `max_z_mm_s2`: Z acceleration limit
- `feeds.rapid_mm_s` (required): Rapid (G0) speed

**New Validators**:
```yaml
- "canvas_mm.x_min >= 0"
- "canvas_mm.x_max <= work_area_mm.x"
- "canvas_mm.y_min >= 0"
- "canvas_mm.y_max <= work_area_mm.y"
- "canvas_mm.x_max > canvas_mm.x_min"
- "canvas_mm.y_max > canvas_mm.y_min"
```

---

## Code Changes

### 1. Validators (`src/utils/validators.py`)

**New Models**:
```python
class CanvasBounds(BaseModel):
    """Canvas bounds within machine work area."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    @validator('x_max')
    def validate_x_range(cls, v, values):
        if v <= values['x_min']:
            raise ValueError("x_max must be > x_min")
        return v

class Acceleration(BaseModel):
    """Acceleration limits (mm/s²)."""
    max_xy_mm_s2: float
    max_z_mm_s2: float
```

**Updated MachineV1**:
```python
class MachineV1(BaseModel):
    # ... existing fields ...
    canvas_mm: CanvasBounds
    acceleration: Acceleration
    
    @validator('canvas_mm')
    def validate_canvas_within_work_area(cls, v, values):
        work = values['work_area_mm']
        if v.x_max > work.x:
            raise ValueError(f"Canvas exceeds machine work area")
        return v
```

**Updated get_stroke_bounds()**:
```python
def get_stroke_bounds(machine_cfg_path):
    machine_cfg = load_machine_profile(machine_cfg_path)
    canvas = machine_cfg.canvas_mm
    
    return {
        'x': (canvas.x_min, canvas.x_max),      # Canvas bounds, not machine!
        'y': (canvas.y_min, canvas.y_max),
        'z': (0.0, machine_cfg.work_area_mm.z),  # Machine Z limit
        'speed': (1.0, machine_cfg.feeds.max_xy_mm_s),
        'cmy': (0.0, 1.0),
    }
```

### 2. G-code Generator (`src/utils/gcode_generator.py`)

**Updated Coordinate Transform**:
```python
def image_mm_to_machine_mm(
    x_img_mm: torch.Tensor,
    canvas_bounds: Tuple[float, float, float, float],  # x_min, x_max, y_min, y_max
    flip_y: bool = True
) -> torch.Tensor:
    """Transform image coords (relative to canvas) to machine coords (absolute)."""
    x_min, x_max, y_min, y_max = canvas_bounds
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    x_mach_mm = x_img_mm.clone()
    
    if flip_y:
        # Flip Y within canvas
        x_mach_mm[..., 1] = canvas_height - x_img_mm[..., 1]
    
    # Add canvas offset to get absolute machine coordinates
    x_mach_mm[..., 0] = x_mach_mm[..., 0] + x_min
    x_mach_mm[..., 1] = x_mach_mm[..., 1] + y_min
    
    return x_mach_mm
```

**Updated linearize_stroke()**:
```python
def linearize_stroke(
    stroke: validators.StrokeV1,
    machine_cfg: validators.MachineV1,  # No longer needs work_area_mm
    max_err_mm: float = 0.5
):
    # ... bezier flattening ...
    
    # Transform using canvas bounds
    canvas = machine_cfg.canvas_mm
    canvas_bounds = (canvas.x_min, canvas.x_max, canvas.y_min, canvas.y_max)
    pts_mach = image_mm_to_machine_mm(pts_img, canvas_bounds, flip_y=True)
    
    # ... rest of function ...
```

**Updated generate_stroke_gcode()**:
```python
def generate_stroke_gcode(
    stroke: validators.StrokeV1,
    machine_cfg: validators.MachineV1,  # No work_area_mm parameter!
    include_purge: bool = True
):
    # Linearize automatically uses canvas bounds from machine_cfg
    pts_mach, z_interp, v_interp = linearize_stroke(stroke, machine_cfg)
    # ... rest ...
```

---

## Remaining Work (TODO)

### A. Update All Call Sites

Search for and update:
```bash
grep -r "generate_stroke_gcode\|generate_cmy_gcode\|generate_pen_gcode" src/ tests/
```

**Remove `work_area_mm` parameters**:
```python
# OLD:
work_area_mm = (machine_cfg.work_area_mm.x, machine_cfg.work_area_mm.y)
generate_stroke_gcode(stroke, machine_cfg, work_area_mm)

# NEW:
generate_stroke_gcode(stroke, machine_cfg)  # Canvas bounds implicit
```

### B. Update G-code VM

**Use acceleration from config**:
```python
# In GCodeVM.__init__:
# OLD:
self.accel_mm_s2 = accel_mm_s2  # Parameter

# NEW:
self.accel_mm_s2 = machine_cfg.acceleration.max_xy_mm_s2  # From config
```

**Check both machine AND canvas limits**:
```python
def check_soft_limits(self, x, y, z, line_idx=None):
    work = self.machine_cfg.work_area_mm
    canvas = self.machine_cfg.canvas_mm
    
    # Check machine physical limits
    if not (0 <= x <= work.x):
        self.violations.append(f"X={x} exceeds MACHINE limit [0, {work.x}]")
    
    # Warn if outside canvas (but allow for purge zones, etc.)
    if not (canvas.x_min <= x <= canvas.x_max):
        logger.warning(f"X={x} outside CANVAS bounds [{canvas.x_min}, {canvas.x_max}]")
```

### C. Update Tests

**Fix test expectations**:
```python
# validate_utils_extended.py

# OLD:
assert bounds['x'] == (0.0, 210.0)

# NEW (canvas can be offset):
assert bounds['x'] == (20.0, 230.0)  # Canvas starts at 20mm
```

**Update coordinate transform tests**:
```python
# Test canvas offset
canvas_bounds = (20.0, 230.0, 26.5, 323.5)
img_coords = torch.tensor([[0.0, 0.0]])  # Top-left of canvas
mach_coords = image_mm_to_machine_mm(img_coords, canvas_bounds)
assert mach_coords[0, 0].item() == 20.0     # Canvas x_min
assert mach_coords[0, 1].item() == 323.5    # Canvas y_max (flipped)
```

### D. Update Pen Layer Generation

```python
# generate_pen_gcode()
def generate_pen_gcode(pen_vectors, machine_cfg, output_path):
    # OLD:
    # work_area_mm = (machine_cfg.work_area_mm.x, machine_cfg.work_area_mm.y)
    # pts_mach = image_mm_to_machine_mm(pts_img, work_area_mm, flip_y=True)
    
    # NEW:
    canvas = machine_cfg.canvas_mm
    canvas_bounds = (canvas.x_min, canvas.x_max, canvas.y_min, canvas.y_max)
    pts_mach = image_mm_to_machine_mm(pts_img, canvas_bounds, flip_y=True)
```

---

## Testing Checklist

- [ ] Load machine config (should validate canvas within work area)
- [ ] Test `get_stroke_bounds()` returns canvas bounds
- [ ] Test coordinate transform with offset canvas
- [ ] Test stroke linearization produces absolute machine coords
- [ ] Test G-code generation with offset canvas
- [ ] Test VM detects violations outside machine limits
- [ ] Test VM warns about off-canvas moves (purge zones)
- [ ] Test acceleration limits from config
- [ ] Run full test suite (comprehensive + extended)

---

## Example Use Cases

### Use Case 1: Centered A4 on Larger Machine
```yaml
work_area_mm:
  x: 250.0      # Machine: 250×350mm
  y: 350.0

canvas_mm:
  x_min: 20.0   # A4 centered: 210mm wide
  x_max: 230.0
  y_min: 26.5   # A4 centered: 297mm tall
  y_max: 323.5
```

Stroke at canvas (0, 0) → Machine (20.0, 323.5)

### Use Case 2: Multiple Paper Formats
```yaml
# Job 1: A4 portrait
canvas_mm: {x_min: 20, x_max: 230, y_min: 26.5, y_max: 323.5}

# Job 2: A3 landscape
canvas_mm: {x_min: 0, x_max: 420, y_min: 0, y_max: 297}
```

### Use Case 3: Off-Canvas Purge Zone
```yaml
canvas_mm:
  x_min: 0
  x_max: 210       # A4 right edge

safety:
  purge_zone_mm:
    x: [220, 240]  # Outside canvas, within machine work area
    y: [290, 310]
```

---

## Benefits

1. **Flexibility**: Support any paper size ≤ machine work area
2. **Safety**: Explicit machine vs canvas bounds
3. **Precision**: Canvas can be precisely positioned (e.g., centered)
4. **Extensibility**: Easy to add job-level canvas overrides
5. **Clarity**: Acceleration and speeds explicit in config

---

## Migration Path for Existing Code

1. **Update machine config** (required):
   - Add `canvas_mm` field
   - Add `acceleration` field
   - Add `feeds.rapid_mm_s` field

2. **Update stroke bounds** (automatic):
   - `get_stroke_bounds()` now returns canvas bounds

3. **Update G-code generation** (function signature change):
   - Remove `work_area_mm` parameters from function calls
   - Canvas bounds implicit from `machine_cfg`

4. **Update tests** (expectations change):
   - Bounds no longer (0, 0) origin
   - Adjust expected coordinate transforms

---

## Files Modified

1. `configs/schema/machine.v1.yaml` - Schema definition
2. `configs/machine_grbl_airbrush_v1.yaml` - Example config
3. `src/utils/validators.py` - Models and validation
4. `src/utils/gcode_generator.py` - Coordinate transforms
5. Tests (TODO) - Update expectations

---

## Next Steps

1. Complete remaining function updates in `gcode_generator.py`
2. Update `gcode_vm.py` to use config acceleration
3. Update all tests
4. Run full validation suite
5. Document canvas positioning guidelines for users

---

**Implementation Date**: 2025-10-30  
**Status**: 60% Complete (Core architecture done, integration remaining)  
**Breaking Changes**: Yes (function signatures, config schema)  
**Migration Required**: Yes (update all machine configs)

