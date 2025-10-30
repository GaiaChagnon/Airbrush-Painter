"""Test LPIPS input normalization and precision.

Validates LPIPS invariants:
    - Inputs are in [-1,1] range
    - Computation runs in FP32 (no autocast)
    - normalize_img_for_lpips() correctly maps [0,1] → [-1,1]
    - LPIPS(img, img) ≈ 0
    - LPIPS(white, black) > threshold

Test cases:
    - test_normalize_img_for_lpips_range()
        * Input [0,1] → output [-1,1] ± 0.01
    - test_lpips_dtype_is_fp32()
        * Verify computation uses float32 tensors
    - test_lpips_same_image_zero()
        * LPIPS(img, img) < 1e-5
    - test_lpips_opposite_images()
        * LPIPS(white, black) > 0.5
    - test_lpips_no_autocast()
        * Verify autocast(enabled=False) in TiledLPIPS.forward()

BF16 bleed check:
    - Wrap in autocast(bf16) and verify LPIPS still uses FP32

Run:
    pytest tests/test_lpips_normalization.py -v
"""

