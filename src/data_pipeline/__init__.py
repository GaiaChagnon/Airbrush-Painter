"""Data preprocessing and calibration for multi-resolution training.

Modules:
    - preprocess: Target image standardization, CMY/pen layer separation, inpainting
    - pen_vectorizer: Binary mask → centerline vectors (Potrace + simplification)
    - calibrate: Manual measurement workflow, LUT generation, validation

Workflow:
    1. Raw image (data/raw_images/) → standardize to render_px
    2. LAB thresholding → pen mask
    3. Inpaint CMY target (color-only)
    4. Vectorize pen mask → YAML paths
    5. Save to data/target_images/

Calibration workflow:
    1. Generate calibration G-code with labeled patterns
    2. Print & scan
    3. Manual measurements → manual_calibration_results.yaml
    4. Interpolate → color_lut.pt, psf_lut.pt, alpha_lut.pt (FP32)
    5. Validate layering model via ΔE comparison

All outputs are deterministic and versioned (schema compliance).
"""

