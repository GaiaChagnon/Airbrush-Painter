"""Inference script: Hybrid loop (Strategist + Technician) → G-code.

Runs full painting pipeline from checkpoint to executable G-code:
    1. Load checkpoint and config
    2. Load/preprocess target image
    3. Run Hybrid loop:
        - Strategist proposes stroke
        - Technician refines stroke (gradient descent)
        - Render stroke on canvas
        - Repeat until stroke_cap reached
    4. Save strokes.yaml (mm-space, validated)
    5. Generate G-code (_cmy.gcode, _pen.gcode)
    6. Write job manifest (_manifest.yaml)
    7. Save final simulated painting
    8. Compute final LPIPS vs. target

Refactored architecture:
    - paint_main(checkpoint_path, job_config, output_dir) → dict
        * Callable function (used by HPO validation loop)
        * Returns: {final_canvas, final_lpips, strokes_path, gcode_path}
    - CLI entry point: if __name__ == "__main__"

Multi-resolution support:
    - Draft mode: Uses render_px from env config
    - Final mode: --print_res_px W H for higher-fidelity rendering
    - Logs all resolutions to MLflow params

Coordinate frames:
    - Internal: image frame (top-left, +Y down) in mm
    - G-code: machine frame (bottom-left, +Y up)
    - Transform happens once in gcode_generator

CLI:
    python scripts/paint.py --checkpoint outputs/checkpoints/best.pth \\
                            --target data/target_images/cmy_only/hard/sample.png \\
                            --output gcode_output/sample/
    python scripts/paint.py --checkpoint best.pth --target sample.png \\
                            --output out/ --print_res_px 1816 2560

Output structure:
    <output_dir>/
        <job_name>_cmy.gcode
        <job_name>_pen.gcode
        <job_name>_manifest.yaml
        <job_name>_strokes.yaml
        <job_name>_final_sim.png

Used by:
    - CLI: Manual painting jobs
    - HPO: Validation loop (paint_main callable)
    - GUI: Inference tab execution
"""

