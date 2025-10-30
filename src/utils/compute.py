"""Numerics, resolution conversions, tiling, and bounds projection.

Core utilities:
    - LPIPS normalization: normalize_img_for_lpips() converts linear RGB [0,1] → [-1,1] FP32
    - Resolution conversions: mm_to_px(), px_to_mm() with frame-aware transforms
    - Coordinate frame transforms: image_mm_to_machine_mm()
    - Tiled inference: TiledLPIPS with Hann windowing for large images
    - Bounds projection: project_to_bounds() for action clamping
    - Memory guards: choose_tile_size_from_mem(), retry_cuda_oom()

Invariants:
    - Internal images are linear RGB [0,1] (F32)
    - LPIPS inputs must be [-1,1] and computed in FP32 (no autocast)
    - Geometry parameters are in mm; conversions happen at boundaries only
    - Work area: (210.0, 297.0) mm (A4 portrait) unless overridden

Multi-resolution architecture:
    - render_px: Physics simulation grid (e.g., 908×1280)
    - obs_px: Policy input (downsampled, e.g., 454×640)
    - reward_px: LPIPS scoring grid (typically == render_px)
"""

