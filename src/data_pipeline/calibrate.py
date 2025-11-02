"""Manual calibration workflow: patterns → measurements → LUTs.

Two-phase calibration:
    Phase 1 - Generate calibration G-code (automated):
        - Read calibration_layout.yaml (grid specs, labels, fiducials)
        - Generate labeled test patterns:
            * Color grid (CMY samples with "Cxyz" labels)
            * PSF/alpha dots (Z/speed samples with "Pxy" labels)
            * Layering patches (overlap tests with "Lxy" labels)
        - Include fiducials for scan alignment
        - Output: gcode_output/calibration/*.gcode

    Phase 2 - Build LUTs from manual measurements (automated):
        - Read manual_calibration_results.yaml (operator-entered data)
        - Interpolate sparse measurements to dense grids (scipy.interpolate):
            * f(C,M,Y) → (R,G,B): Trilinear on CMY cube
            * g(Z,V) → Alpha: Bilinear on Z×V grid
            * h(Z,V) → PSF: Gaussian kernel from width_mm
        - Save to configs/sim/luts/{color,alpha,psf}_lut.pt (FP32)
        - Validate layering model: Predict overlap color, compare ΔE vs. measured

Public API:
    generate_calibration_gcode(machine_cfg, calib_layout_cfg, out_dir)
        → Saves G-code files via src.utils.fs.atomic_write_bytes
    build_luts_from_manual(manual_results_yaml, out_dir)
        → {"color": path, "alpha": path, "psf": path}

Manual measurement workflow (operator):
    1. Print calibration G-code
    2. Scan at known DPI → data/calibration_scans/
    3. Measure in calibrated editor (linear RGB, mm ruler)
    4. Enter data into manual_calibration_results.yaml
    5. Run build_luts_from_manual()
"""

