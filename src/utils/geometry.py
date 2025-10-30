"""Geometric operations for strokes and paths.

Provides:
    - Cubic Bézier evaluation and adaptive flattening
    - Polyline operations: length, bbox, arc-length parametrization
    - Clipping polylines to rectangular regions
    - Stroke bounding box calculation for tile culling

Used by:
    - Renderer: Bézier → polyline ribbon for rasterization
    - G-code generator: Linearization with interpolated Z/speed
    - Environment: Tile culling (only render overlapped tiles)
    - Tests: Synthetic stroke generation and validation

All coordinates in millimeters (mm) unless explicitly noted as pixels.
Conversions to pixels happen at renderer boundaries via compute.mm_to_px().

Adaptive flattening uses recursive subdivision with configurable max_err_px
tolerance (default: 0.25 px for sub-pixel accuracy).
"""

