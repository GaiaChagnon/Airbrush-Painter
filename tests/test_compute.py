"""Test numerics and tiling utilities.

Tests for src.utils.compute:
    - Tiled LPIPS ≈ full-frame LPIPS (on small images)
    - Hann window properties (smooth, unit norm)
    - project_to_bounds idempotent and correct
    - normalize_img_for_lpips output range [-1,1]
    - BF16/FP32 parity for LPIPS (within tolerance)

Test cases:
    - test_tiled_lpips_parity()
    - test_hann2d_properties()
    - test_project_to_bounds_idempotent()
    - test_normalize_img_for_lpips_range()
    - test_bf16_fp32_lpips_parity()
    - test_apply_tiled_identity()

Property tests:
    - apply_tiled(fn=identity) ≈ input
    - project_to_bounds(project_to_bounds(x)) == project_to_bounds(x)

Run:
    pytest tests/test_compute.py -v
"""

