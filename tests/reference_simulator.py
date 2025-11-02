"""Reference simulator for unit tests and validation.

Simple, CPU-only renderer for:
    - Unit test golden outputs
    - Quick debugging without GPU
    - Sanity checks against GPU path

Implementation:
    - Pure NumPy/PIL (no CUDA)
    - Bézier → polyline (adaptive flattening)
    - Gaussian stamp rasterization (no nvdiffrast)
    - Alpha-over compositing
    - Same LUTs as production renderer

Public API:
    render_stroke(canvas_np, params_mm_np) → canvas_np

Used by:
    - test_renderer.py: Validate GPU vs. CPU parity
    - test_mm_px_roundtrip.py: Coordinate conversion checks
    - Synthetic stroke generation for other tests

NOT imported by training or inference (tests only).
Intentionally simple (no optimization, no batching).
"""

