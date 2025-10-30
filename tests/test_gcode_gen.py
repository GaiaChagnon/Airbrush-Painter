"""Test G-code generation correctness.

Tests for src.utils.gcode_generator:
    - Soft-limit violations detected
    - Feed conversions: mm/s → F<mm/min>
    - Stroke markers present: ; STROKE_ID: <id>
    - Macros inserted: PURGE, PEN_UP, PEN_DOWN
    - GRBL-safe header (no RepRap codes like M82)
    - Coordinate frame transform: image → machine

Test cases:
    - test_soft_limit_violation_detected()
    - test_feed_conversion_correct()
    - test_stroke_markers_present()
    - test_macros_inserted()
    - test_grbl_safe_header()
    - test_coordinate_frame_transform()

Corner stroke test:
    - Stroke at (0,0) in image → (0, work_area_mm.y) in G-code
    - Stroke at (work_area_mm.x, work_area_mm.y) → (work_area_mm.x, 0)

Round-trip test:
    - Generate G-code → parse back → verify matches original strokes

Run:
    pytest tests/test_gcode_gen.py -v
"""

