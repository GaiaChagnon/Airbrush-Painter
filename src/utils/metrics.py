"""Image quality metrics for CI and adversarial tests.

Provides:
    - PSNR: Peak signal-to-noise ratio
    - SSIM: Structural similarity index
    - paint_coverage(): Fraction of canvas painted (for reward-hack detection)
    - edge_preservation_score(): Validates structure preservation

Used by:
    - Golden CI tests: PSNR/SSIM thresholds alongside LPIPS
    - Adversarial tests: Detect background-flooding exploits
    - HPO validation: Multi-metric evaluation

All metrics operate on torch tensors (3, H, W) or (B, 3, H, W) in linear RGB [0,1].
Returns scalar tensors (reduced over spatial/batch dims).

LPIPS (primary reward metric) is in compute.TiledLPIPS, not here.
"""

