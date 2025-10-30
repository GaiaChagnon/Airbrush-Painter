"""Target image preprocessing: standardization, layer separation, inpainting.

Converts user-provided images into training-ready targets:
    1. Standardize resolution to render_px (e.g., 908×1280, A4 portrait)
    2. Maintain aspect ratio via padding/cropping
    3. LAB color space thresholding to detect pen mask (L < threshold)
    4. Morphological cleanup (closing, opening) to remove noise
    5. Inpaint CMY target using cv2.INPAINT_TELEA or INPAINT_NS
    6. Save color-only PNG to data/target_images/cmy_only/{easy,medium,hard}/
    7. Pass pen mask to pen_vectorizer for path extraction

Public API:
    preprocess_image(raw_image_path, output_dir_cmy, output_dir_pen,
                     target_render_px, pen_threshold_lab_l=10)
        → (cmy_png_path, pen_yaml_path)

Outputs:
    - CMY target: Linear RGB [0,1], 8-bit PNG, resolution == render_px
    - Pen mask: Binary, passed to pen_vectorizer

Used by:
    - Training: Preprocess entire dataset before training
    - Inference: Prepare custom targets
    - Validation: Create fixed validation set
"""

