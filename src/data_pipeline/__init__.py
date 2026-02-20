"""Data preprocessing and calibration for multi-resolution training.

Modules:
    - preprocess: Target image standardization, CMY layer separation, inpainting
    - calibrate: Manual measurement workflow, LUT generation, validation

Calibration workflow:
    1. Generate calibration G-code with labeled patterns
    2. Print & scan
    3. Manual measurements → manual_calibration_results.yaml
    4. Interpolate → color_lut.pt, psf_lut.pt, alpha_lut.pt (FP32)
    5. Validate layering model via ΔE comparison

All outputs are deterministic and versioned (schema compliance).
"""

