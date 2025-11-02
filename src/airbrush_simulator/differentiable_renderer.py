"""Production differentiable renderer with hardened fallback strategy.

Implements physics-based paint rendering with learned LUTs:
    - Bézier curve → polyline ribbon (adaptive flattening)
    - CUDA rasterization (nvdiffrast) → alpha mask
    - CMY → RGB via color LUT (trilinear interpolation)
    - Alpha modulation via alpha_lut(z, speed)
    - PSF convolution or conditioned rasterization
    - Alpha-over compositing on canvas

Public API:
    class DifferentiableRenderer:
        render_stroke_tiled(canvas, params_mm) → canvas_new
        fine_tune_stroke_technician(a_init_mm, canvas, target) → a_refined_mm
        project_params(params_mm) → params_mm_clamped
        load_luts(color_lut_path, alpha_lut_path, psf_lut_path)

Fallback strategy:
    - render_stroke_tiled (env path): CUDA error → log + splat fallback → proceed
    - fine_tune_stroke_technician (inference path): CUDA error → raise (no fallback)
    Rationale: Don't change optimization landscape mid-refinement

Parameter projection (mm-space):
    - XY: [0, work_area_mm]
    - Z: [z_min, z_max] from schema
    - Speed: [v_min, v_max] from schema
    - CMY: [0.0, 1.0]

Technician refinement:
    - N gradient descent steps (configurable, e.g., 12)
    - Early stop: ΔLPIPS < threshold for K consecutive steps
    - Projects params each step to maintain validity

Precision:
    - LUTs: FP32 (stability)
    - Forward/backward: BF16 autocast (DGX Spark)
    - LPIPS: FP32 (accuracy)
"""

