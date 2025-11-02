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

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

from src.data_pipeline import pen_tracer
from src.utils import validators, fs, gcode_generator

logger = logging.getLogger(__name__)


def paint_main(
    checkpoint_path: str,
    target_image_path: str,
    output_dir: str,
    env_cfg_path: str = "configs/env_airbrush_v1.yaml",
    machine_cfg_path: str = "configs/machine_grbl_airbrush_v1.yaml",
    pen_tool_cfg_path: str = "configs/tools/pen_finetip_v1.yaml",
    pen_tracer_cfg_path: str = "configs/sim/pen_tracer_v2.yaml",
    enable_pen_layer: bool = True,
    print_res_px: Optional[tuple] = None,
) -> Dict[str, any]:
    """Main painting pipeline with optional pen layer.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to trained model checkpoint
    target_image_path : str
        Path to target image (standardized at render_px)
    output_dir : str
        Output directory for artifacts
    env_cfg_path : str
        Path to environment config
    machine_cfg_path : str
        Path to machine profile
    pen_tool_cfg_path : str
        Path to pen tool config
    pen_tracer_cfg_path : str
        Path to pen tracer config
    enable_pen_layer : bool
        Enable pen layer generation, default True
    print_res_px : Optional[tuple]
        Optional (W, H) for final print resolution (higher than render_px)
    
    Returns
    -------
    Dict[str, any]
        Results dict with:
            - final_canvas: torch.Tensor (simulated CMY canvas)
            - final_lpips: float
            - strokes_path: str (path to strokes.yaml)
            - cmy_gcode_path: str
            - pen_gcode_path: Optional[str]
            - manifest_path: str
            - pen_preview_path: Optional[str]
            - composite_path: Optional[str]
    
    Notes
    -----
    This is a stub implementation showing pen tracer integration points.
    Full implementation would include:
        1. Load checkpoint and run Hybrid loop (Strategist + Technician)
        2. Generate strokes.yaml
        3. Render final CMY canvas
        4. Call pen_tracer.make_pen_layer() if enable_pen_layer=True
        5. Generate G-code for both CMY and pen layers
        6. Save all artifacts
    """
    logger.info(f"Starting paint pipeline: {target_image_path}")
    
    out_path = Path(output_dir)
    fs.ensure_dir(out_path)
    
    # Load configs
    machine_cfg = validators.load_machine_profile(machine_cfg_path)
    
    # =======================================================================
    # CMY PAINTING PASS
    # =======================================================================
    # TODO: Implement full Hybrid loop (Strategist + Technician)
    # This would include:
    #   1. Load checkpoint
    #   2. Initialize env and renderer
    #   3. Run painting loop (stroke_cap iterations)
    #   4. Save strokes.yaml
    #   5. Render final CMY canvas
    
    # Placeholder: Assume CMY painting completed
    logger.info("CMY painting pass complete (placeholder)")
    
    # For now, create placeholder paths
    job_name = Path(target_image_path).stem
    strokes_yaml_path = out_path / f"{job_name}_strokes.yaml"
    cmy_gcode_path = out_path / f"{job_name}_cmy.gcode"
    final_sim_path = out_path / f"{job_name}_final_sim.png"
    
    # =======================================================================
    # PEN LAYER GENERATION (NEW)
    # =======================================================================
    pen_gcode_path = None
    pen_preview_path = None
    composite_path = None
    
    if enable_pen_layer:
        logger.info("Generating pen layer...")
        
        try:
            # Generate pen layer from target image
            pen_result = pen_tracer.make_pen_layer(
                target_rgb_path=target_image_path,
                env_cfg_path=env_cfg_path,
                pen_tool_cfg_path=pen_tool_cfg_path,
                pen_tracer_cfg_path=pen_tracer_cfg_path,
                out_dir=str(out_path / "pen"),
                cmy_canvas_path=str(final_sim_path) if final_sim_path.exists() else None,
            )
            
            # Extract paths
            pen_vectors_yaml_path = pen_result['pen_vectors_yaml']
            pen_preview_path = pen_result['pen_preview_png']
            composite_path = pen_result['composite_png']
            
            # Load validated pen vectors
            pen_vectors = validators.load_pen_vectors(pen_vectors_yaml_path)
            pen_tool_cfg = validators.load_pen_tool_config(pen_tool_cfg_path)
            
            # Generate pen G-code
            pen_gcode_path = out_path / f"{job_name}_pen.gcode"
            gcode_generator.generate_pen_gcode(
                pen_vectors,
                machine_cfg,
                pen_tool_cfg,
                pen_gcode_path,
            )
            
            logger.info(f"Pen layer complete: {pen_gcode_path}")
            logger.info(f"Pen metrics: {pen_result['metrics']}")
            
        except Exception as e:
            logger.error(f"Pen layer generation failed: {e}")
            logger.warning("Continuing without pen layer")
    
    # =======================================================================
    # JOB MANIFEST
    # =======================================================================
    manifest_path = out_path / f"{job_name}_manifest.yaml"
    
    # Create minimal job manifest (placeholder)
    manifest_data = {
        'schema': 'job.v1',
        'machine_profile': machine_cfg_path,
        'inputs': {
            'target_image_path': target_image_path,
            'pen_layer_path': pen_vectors_yaml_path if enable_pen_layer else None,
        },
        'artifacts': {
            'cmy_gcode': str(cmy_gcode_path),
            'pen_gcode': str(pen_gcode_path) if pen_gcode_path else None,
            'manifest': str(manifest_path),
            'strokes_yaml': str(strokes_yaml_path),
            'final_sim': str(final_sim_path),
            'pen_preview': pen_preview_path,
            'composite': composite_path,
        },
    }
    
    fs.atomic_yaml_dump(manifest_data, manifest_path)
    
    logger.info(f"Painting complete. Artifacts saved to {output_dir}")
    
    return {
        'final_canvas': None,  # TODO: Return actual canvas tensor
        'final_lpips': 0.0,  # TODO: Compute actual LPIPS
        'strokes_path': str(strokes_yaml_path),
        'cmy_gcode_path': str(cmy_gcode_path),
        'pen_gcode_path': str(pen_gcode_path) if pen_gcode_path else None,
        'manifest_path': str(manifest_path),
        'pen_preview_path': pen_preview_path,
        'composite_path': composite_path,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paint target image using trained model and generate G-code"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target image (PNG/JPEG)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env_airbrush_v1.yaml",
        help="Path to environment config",
    )
    parser.add_argument(
        "--machine-config",
        type=str,
        default="configs/machine_grbl_airbrush_v1.yaml",
        help="Path to machine config",
    )
    parser.add_argument(
        "--pen-tool-config",
        type=str,
        default="configs/tools/pen_finetip_v1.yaml",
        help="Path to pen tool config",
    )
    parser.add_argument(
        "--pen-tracer-config",
        type=str,
        default="configs/sim/pen_tracer_v2.yaml",
        help="Path to pen tracer config",
    )
    parser.add_argument(
        "--no-pen-layer",
        action="store_true",
        help="Disable pen layer generation (CMY only)",
    )
    parser.add_argument(
        "--print-res-px",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Optional print resolution (W H) for final rendering",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # Run pipeline
    result = paint_main(
        checkpoint_path=args.checkpoint,
        target_image_path=args.target,
        output_dir=args.output,
        env_cfg_path=args.env_config,
        machine_cfg_path=args.machine_config,
        pen_tool_cfg_path=args.pen_tool_config,
        pen_tracer_cfg_path=args.pen_tracer_config,
        enable_pen_layer=not args.no_pen_layer,
        print_res_px=tuple(args.print_res_px) if args.print_res_px else None,
    )
    
    print("\n=== Paint Complete ===")
    print(f"CMY G-code: {result['cmy_gcode_path']}")
    if result['pen_gcode_path']:
        print(f"Pen G-code: {result['pen_gcode_path']}")
    print(f"Manifest: {result['manifest_path']}")
    if result['composite_path']:
        print(f"Composite preview: {result['composite_path']}")
    

if __name__ == "__main__":
    main()
