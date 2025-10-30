"""Test image quality metrics.

Tests for src.utils.metrics:
    - PSNR: Identical images → inf, opposite images → low
    - SSIM: Identical images → 1.0, opposite images → low
    - paint_coverage(): Monotonic with alpha
    - edge_preservation_score(): High for preserved edges

Test cases:
    - test_psnr_identical_images()
    - test_psnr_opposite_images()
    - test_ssim_identical_images()
    - test_ssim_opposite_images()
    - test_paint_coverage_monotonic()
    - test_edge_preservation_score()

Known values:
    - PSNR(img, img) → inf (or very high with epsilon)
    - SSIM(img, img) → 1.0 ± 1e-6

Coverage test:
    - Blank canvas: coverage ≈ 0
    - Fully painted canvas: coverage ≈ 1

Run:
    pytest tests/test_metrics.py -v
"""

