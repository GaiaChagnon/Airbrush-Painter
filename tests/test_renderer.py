"""Test differentiable renderer correctness and robustness.

Tests for src.airbrush_simulator.differentiable_renderer:
    - Parameter projection (mm-space clamping)
    - No NaNs under fuzzed actions (1000 trials)
    - Fallback path (splat) parity with primary (within tolerance)
    - Technician raises on CUDA error (no fallback in inference)
    - LUT interpolation correctness
    - Alpha-over compositing (associativity check)

Test cases:
    - test_project_params_clamps_to_bounds()
    - test_render_stroke_no_nans_fuzz()
    - test_fallback_splat_parity()
    - test_technician_raises_on_cuda_error()
    - test_lut_interpolation_smooth()
    - test_alpha_over_compositing()

Synthetic strokes:
    - Corner strokes (0,0), (work_area_mm.x, work_area_mm.y)
    - Out-of-bounds strokes (projection test)
    - Degenerate Béziers (straight lines, points)

Run:
    pytest tests/test_renderer.py -v --tb=short
"""

