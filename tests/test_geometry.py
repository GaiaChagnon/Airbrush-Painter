"""Test geometric operations for strokes.

Tests for src.utils.geometry:
    - Cubic Bézier evaluation at t=0,0.5,1
    - Adaptive flattening converges with max_err_px
    - Polyline length calculation
    - Bounding box correctness
    - Arc-length parametrization
    - Polyline clipping to rect

Test cases:
    - test_bezier_cubic_eval()
    - test_bezier_cubic_polyline_convergence()
    - test_polyline_length()
    - test_polyline_bbox()
    - test_parametrize_by_arclength()
    - test_clip_polyline_to_rect()

Degenerate cases:
    - Straight line (control points collinear)
    - Single point (all control points identical)
    - Out-of-bounds Bézier (clipping test)

Convergence test:
    - max_err_px ↓ → polyline_length converges

Run:
    pytest tests/test_geometry.py -v
"""

