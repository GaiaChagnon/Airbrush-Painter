"""G-code generation from stroke and pen vectors.

Converts validated strokes (stroke.v1.yaml) and pen paths to GRBL-compatible G-code:
    - Coordinate frame transform: image (top-left, +Y down) → machine (bottom-left, +Y up)
    - Bézier linearization with interpolated Z and speed
    - Feed conversion: mm/s → F<mm/min>
    - Macro insertion: PURGE before each stroke, PEN_UP/PEN_DOWN for pen layer
    - Soft-limit validation against machine profile
    - Stroke ID comments: ; STROKE_ID: <id>

Outputs:
    <job>_cmy.gcode: Airbrush painting pass
    <job>_pen.gcode: Black pen layer
    <job>_manifest.yaml: Complete job provenance (job.v1.yaml schema)

GRBL-safe header (no RepRap codes):
    G21            ; mm units
    G90            ; absolute positioning
    G92 X0 Y0 Z0   ; set current position as origin (optional)

Machine-specific post-processors via strategy pattern (extensible).
Validates all moves against work_area_mm and feeds.max_xy_mm_s.

Usage:
    from src.utils import gcode_generator, fs
    gcode_generator.generate_all_gcode(...)
    # Files written atomically via fs.atomic_write_bytes()
"""

