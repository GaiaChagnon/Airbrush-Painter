"""Color space conversions and perceptual metrics.

Provides:
    - sRGB ↔ linear RGB conversions (gamma 2.2 approximation)
    - RGB → Lab color space (CIE L*a*b*, D65 illuminant)
    - ΔE2000: Perceptual color difference (CIEDE2000 formula)
    - Luminance calculation from linear RGB

Used by:
    - Calibration pipeline: ΔE validation of layering model
    - Preprocessing: LAB thresholding for pen mask extraction
    - Metrics: Perceptual color accuracy assessment

All conversions operate on torch tensors (3, H, W) or (B, 3, H, W).
Input/output ranges are documented per function.

Invariants:
    - Internal pipeline uses linear RGB [0,1]
    - sRGB [0,1] only at I/O boundaries (load/save)
    - Lab coordinates: L[0,100], a,b[-128,127]
"""

