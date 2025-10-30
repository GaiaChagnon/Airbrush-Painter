"""Test coordinate frame transformations.

Validates image frame (top-left, +Y down) → machine frame (bottom-left, +Y up):
    - Corner strokes transform correctly
    - Roundtrip: image_mm → machine_mm → G-code → parsed → matches original
    - Renderer uses image frame internally
    - G-code generator applies transform once

Test cases:
    - test_corner_stroke_top_left()
        * Image (0, 0) → Machine (0, work_area_mm.y)
    - test_corner_stroke_bottom_right()
        * Image (work_area_mm.x, work_area_mm.y) → Machine (work_area_mm.x, 0)
    - test_center_stroke_invariant()
        * Center coordinate transforms correctly
    - test_gcode_roundtrip()
        * Generate G-code → parse → verify coordinates

Frame specification (env config):
    coordinate_frames:
      image_origin: "top_left"
      machine_origin: "bottom_left"
      flip_y_for_gcode: true

Run:
    pytest tests/test_coordinate_frames.py -v
"""

