# Pen Tracer Critical Fixes Implementation Plan

**Date:** 2025-11-01  
**Status:** Implementation in progress

---

## Summary of Required Fixes

Based on code review, the following critical issues must be addressed:

### ✅ Completed
1. Import fs from io module for consistency

### 🔧 In Progress
2. Fix anisotropic mm↔px scaling (use sx and sy separately)
3. Add .cpu() before .numpy() for GPU safety  
4. Implement hue_ranges filtering in gamut mask
5. Use proper linear→sRGB conversion for Canny edge detection
6. Fix PenVectors metadata to match schema
7. Fix coverage calculation to account for pen tip width
8. Enforce min_line_spacing_mm in hatch generation
9. Dilate edge mask to prevent hatch overlap
10. Use actual pen tip width for preview rendering

---

## Implementation Notes

Due to the extensive nature of these fixes (affecting ~15 locations across 600+ lines), I recommend:

**Option A: Incremental Fixes (Current Approach)**
- Fix each issue one at a time
- Test after each change
- More conservative, easier to debug

**Option B: Comprehensive Rewrite**
- Create pen_tracer_fixed.py with all corrections
- Test thoroughly
- Replace original once validated

**Recommendation:** Option B for efficiency, given the number of interconnected changes.

---

## Key Changes Required

### 1. Anisotropic Scaling
```python
# OLD (line 409):
mm_per_px = work_area_mm[1] / H

# NEW:
sx_mm_per_px = work_area_mm[0] / W  # X scale
sy_mm_per_px = work_area_mm[1] / H  # Y scale

def px_to_mm_arr(pts_px: np.ndarray) -> np.ndarray:
    return np.stack([pts_px[:,0] * sx_mm_per_px,
                     pts_px[:,1] * sy_mm_per_px], axis=1)
```

### 2. GPU Safety
```python
# Add .detach().cpu() before all .numpy() calls
L = img_lab[0].detach().cpu().numpy()
a = img_lab[1].detach().cpu().numpy()
b = img_lab[2].detach().cpu().numpy()
```

### 3. Schema-Compliant Metadata
```python
# Collect all hatch angles used
all_angles = sorted({a for lvl in pen_tracer_cfg.shadow_hatching.darkness_levels
                       for a in lvl.hatch_angles})

metadata = {
    'tool_name': pen_tool_cfg.name,
    'offset_mm': pen_tool_cfg.offset_mm,
    'hatch_angles_deg': all_angles,  # REQUIRED
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'tracer_version': 'pen_tracer.v2'  # Match schema version
}
```

### 4. Coverage with Thickness
```python
thick_px = max(1, int(round(pen_tool_cfg.tip_diameter_mm / sy_mm_per_px)))
line_length_px = line_length_mm / sy_mm_per_px
line_area_px = int(np.ceil(line_length_px * thick_px))
current_hatch_pixels += line_area_px
```

### 5. Edge-Hatch Exclusion
```python
if pen_tracer_cfg.edge_detection.enabled and edge_mask is not None:
    dilation_px = max(1, int(0.5 * pen_tool_cfg.tip_diameter_mm / sy_mm_per_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
    edge_fat = cv2.dilate(edge_mask, kernel)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(edge_fat))
```

---

## Testing Plan

After fixes:
1. Run `run_pen_tracer_test.py` - verify no crashes
2. Check YAML validates against schema
3. Verify coverage metric matches visual preview
4. Confirm no edge-hatch overlap in output
5. Test with non-A4 aspect ratio image

---

## Status

**Current:** Identified all issues, started incremental fixes  
**Next:** Complete remaining 9 critical fixes  
**ETA:** ~30 minutes for comprehensive implementation

