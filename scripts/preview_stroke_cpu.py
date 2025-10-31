#!/usr/bin/env python3
"""CPU stroke preview tool for visual validation.

CLI tool to render strokes using the CPU reference renderer and generate
diagnostic visualizations: rendered images, alpha maps, PSF profiles, and
cross-sections.

Usage:
    # From YAML file
    python scripts/preview_stroke_cpu.py --stroke_file strokes.yaml --output_dir outputs/preview
    
    # Single stroke with inline parameters
    python scripts/preview_stroke_cpu.py \
        --x1 50 --y1 50 --x2 70 --y2 80 --x3 130 --y3 120 --x4 150 --y4 150 \
        --z0 10 --z1 15 --v0 50 --v1 100 \
        --c 0.8 --m 0.2 --y 0.1 \
        --output_dir outputs/preview
    
    # Multiple strokes from YAML
    python scripts/preview_stroke_cpu.py --stroke_file epoch_042_strokes.yaml --output_dir outputs/epoch_042

Outputs:
    - stroke_render.png: sRGB rendered image
    - alpha_map.png: Grayscale alpha coverage
    - psf_profiles.png: Radial PSF profiles at sample points
    - cross_sections.csv: Line width measurements
    - metadata.yaml: Rendering metadata (time, params, etc.)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.airbrush_simulator.cpu_reference import CPUReferenceRenderer, load_toy_luts
from src.utils import fs, color as color_utils, strokes as stroke_utils, logging_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preview strokes using CPU reference renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--stroke_file',
        type=str,
        help='Path to YAML file containing strokes'
    )
    input_group.add_argument(
        '--inline',
        action='store_true',
        help='Use inline stroke parameters (see --x1, --y1, etc.)'
    )
    
    # Inline stroke parameters (Bézier control points)
    parser.add_argument('--x1', type=float, help='Control point 1 x (mm)')
    parser.add_argument('--y1', type=float, help='Control point 1 y (mm)')
    parser.add_argument('--x2', type=float, help='Control point 2 x (mm)')
    parser.add_argument('--y2', type=float, help='Control point 2 y (mm)')
    parser.add_argument('--x3', type=float, help='Control point 3 x (mm)')
    parser.add_argument('--y3', type=float, help='Control point 3 y (mm)')
    parser.add_argument('--x4', type=float, help='Control point 4 x (mm)')
    parser.add_argument('--y4', type=float, help='Control point 4 y (mm)')
    
    # Z and speed profiles
    parser.add_argument('--z0', type=float, help='Height at start (mm)')
    parser.add_argument('--z1', type=float, help='Height at end (mm)')
    parser.add_argument('--v0', type=float, help='Speed at start (mm/s)')
    parser.add_argument('--v1', type=float, help='Speed at end (mm/s)')
    
    # Color (CMY)
    parser.add_argument('--c', type=float, help='Cyan component [0,1]')
    parser.add_argument('--m', type=float, help='Magenta component [0,1]')
    parser.add_argument('--y', type=float, help='Yellow component [0,1]')
    
    # Canvas settings
    parser.add_argument(
        '--canvas_size',
        type=str,
        default='908,1280',
        help='Canvas size in pixels (H,W), default: 908,1280'
    )
    parser.add_argument(
        '--work_area',
        type=str,
        default='210.0,297.0',
        help='Work area in mm (W,H), default: 210.0,297.0 (A4)'
    )
    parser.add_argument(
        '--background',
        type=str,
        default='white',
        choices=['white', 'black', 'gray'],
        help='Background color, default: white'
    )
    
    # Output settings
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/preview_cpu',
        help='Output directory, default: outputs/preview_cpu'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='stroke',
        help='Output filename prefix, default: stroke'
    )
    
    # Diagnostic outputs
    parser.add_argument(
        '--psf_samples',
        type=int,
        default=5,
        help='Number of PSF profile samples, default: 5'
    )
    parser.add_argument(
        '--cross_sections',
        type=int,
        default=3,
        help='Number of cross-sections for width measurement, default: 3'
    )
    parser.add_argument(
        '--no_profiles',
        action='store_true',
        help='Skip PSF profile generation'
    )
    
    # LUT options
    parser.add_argument(
        '--lut_dir',
        type=str,
        default=None,
        help='Directory with calibrated LUTs (if None, use toy LUTs)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_strokes_from_file(path: Path) -> List[Dict]:
    """Load strokes from YAML file.
    
    Parameters
    ----------
    path : Path
        Path to strokes YAML file
    
    Returns
    -------
    list of dict
        List of stroke dictionaries
    """
    data = fs.load_yaml(path)
    if 'strokes' in data:
        return data['strokes']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid strokes file format: {path}")


def create_inline_stroke(args) -> Dict:
    """Create stroke dictionary from inline arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns
    -------
    dict
        Stroke dictionary
    """
    required_params = [
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
        'z0', 'z1', 'v0', 'v1', 'c', 'm', 'y'
    ]
    
    missing = [p for p in required_params if getattr(args, p) is None]
    if missing:
        raise ValueError(
            f"Missing required parameters for inline stroke: {missing}"
        )
    
    stroke = {
        'id': 'inline-0',
        'bezier': {
            'p1': (args.x1, args.y1),
            'p2': (args.x2, args.y2),
            'p3': (args.x3, args.y3),
            'p4': (args.x4, args.y4)
        },
        'z_profile': {
            'z0': args.z0,
            'z1': args.z1
        },
        'speed_profile': {
            'v0': args.v0,
            'v1': args.v1
        },
        'color_cmy': {
            'c': args.c,
            'm': args.m,
            'y': args.y
        }
    }
    
    return stroke


def get_background_color(name: str) -> np.ndarray:
    """Get background color as linear RGB.
    
    Parameters
    ----------
    name : str
        Color name: 'white', 'black', 'gray'
    
    Returns
    -------
    np.ndarray
        Linear RGB color, shape (3,)
    """
    if name == 'white':
        return np.ones(3, dtype=np.float32)
    elif name == 'black':
        return np.zeros(3, dtype=np.float32)
    elif name == 'gray':
        return np.full(3, 0.5, dtype=np.float32)
    else:
        raise ValueError(f"Unknown background color: {name}")


def save_rgb_image(canvas: np.ndarray, path: Path):
    """Save canvas as sRGB PNG.
    
    Parameters
    ----------
    canvas : np.ndarray
        Canvas in linear RGB [0,1], shape (H, W, 3)
    path : Path
        Output path
    """
    # Convert to sRGB (convert to torch first)
    import torch
    canvas_torch = torch.from_numpy(canvas)
    canvas_srgb_torch = color_utils.linear_to_srgb(canvas_torch)
    canvas_srgb = canvas_srgb_torch.numpy()
    
    # Convert to uint8
    canvas_uint8 = np.clip(canvas_srgb * 255, 0, 255).astype(np.uint8)
    
    # Save
    import PIL.Image
    img = PIL.Image.fromarray(canvas_uint8, mode='RGB')
    img.save(path)


def save_alpha_map(alpha: np.ndarray, path: Path):
    """Save alpha map as grayscale PNG.
    
    Parameters
    ----------
    alpha : np.ndarray
        Alpha map [0,1], shape (H, W)
    path : Path
        Output path
    """
    alpha_uint8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    
    import PIL.Image
    img = PIL.Image.fromarray(alpha_uint8, mode='L')
    img.save(path)


def plot_psf_profiles(
    renderer: CPUReferenceRenderer,
    strokes: List[Dict],
    num_samples: int,
    output_path: Path
):
    """Plot radial PSF profiles at sample points along strokes.
    
    Parameters
    ----------
    renderer : CPUReferenceRenderer
        Renderer instance (for LUT access)
    strokes : list of dict
        Stroke definitions
    num_samples : int
        Number of sample points per stroke
    output_path : Path
        Output path for plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect sample points
    z_samples = []
    v_samples = []
    sigma_samples = []
    alpha_samples = []
    
    for stroke in strokes[:3]:  # Limit to first 3 strokes
        z0 = stroke['z_profile']['z0']
        z1 = stroke['z_profile']['z1']
        v0 = stroke['speed_profile']['v0']
        v1 = stroke['speed_profile']['v1']
        
        t_vals = np.linspace(0, 1, num_samples)
        for t in t_vals:
            z = z0 + t * (z1 - z0)
            v = v0 + t * (v1 - v0)
            
            sigma = renderer._interpolate_psf_lut(z, v)
            alpha = renderer._interpolate_alpha_lut(z, v)
            
            z_samples.append(z)
            v_samples.append(v)
            sigma_samples.append(sigma)
            alpha_samples.append(alpha)
    
    # Plot σ(z, v)
    sc1 = axes[0].scatter(
        z_samples, v_samples, c=sigma_samples,
        cmap='viridis', s=50, alpha=0.7
    )
    axes[0].set_xlabel('z (mm)')
    axes[0].set_ylabel('v (mm/s)')
    axes[0].set_title('PSF Width σ (pixels)')
    plt.colorbar(sc1, ax=axes[0])
    
    # Plot α(z, v)
    sc2 = axes[1].scatter(
        z_samples, v_samples, c=alpha_samples,
        cmap='plasma', s=50, alpha=0.7
    )
    axes[1].set_xlabel('z (mm)')
    axes[1].set_ylabel('v (mm/s)')
    axes[1].set_title('Alpha Coverage α')
    plt.colorbar(sc2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging_config.setup_logging(log_level=log_level, log_file=None)
    logger = logging.getLogger(__name__)
    
    # Parse canvas settings
    canvas_h, canvas_w = map(int, args.canvas_size.split(','))
    work_w, work_h = map(float, args.work_area.split(','))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    fs.ensure_dir(output_dir)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Canvas: {canvas_h}×{canvas_w} px, Work area: {work_w}×{work_h} mm")
    
    # Load or create strokes
    if args.stroke_file:
        stroke_file = Path(args.stroke_file)
        logger.info(f"Loading strokes from: {stroke_file}")
        strokes = load_strokes_from_file(stroke_file)
    else:
        logger.info("Creating inline stroke")
        strokes = [create_inline_stroke(args)]
    
    logger.info(f"Rendering {len(strokes)} stroke(s)")
    
    # Load LUTs
    if args.lut_dir:
        lut_dir = Path(args.lut_dir)
        logger.info(f"Loading calibrated LUTs from: {lut_dir}")
        # TODO: Implement calibrated LUT loading
        raise NotImplementedError("Calibrated LUT loading not yet implemented")
    else:
        logger.info("Using toy LUTs (uncalibrated)")
        luts = load_toy_luts()
    
    # Create renderer
    env_cfg = {
        'work_area_mm': (work_w, work_h),
        'render_px': (canvas_h, canvas_w)
    }
    sim_cfg = {}  # Empty for now
    
    renderer = CPUReferenceRenderer(sim_cfg, env_cfg, luts)
    
    # Initialize canvas
    bg_color = get_background_color(args.background)
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.float32)
    alpha_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    # Render strokes
    start_time = time.time()
    canvas, alpha_map = renderer.render_strokes(canvas, alpha_map, strokes)
    render_time = time.time() - start_time
    
    logger.info(f"Rendering completed in {render_time:.3f}s")
    
    # Save outputs
    prefix = args.prefix
    
    # Rendered image
    render_path = output_dir / f'{prefix}_render.png'
    save_rgb_image(canvas, render_path)
    logger.info(f"Saved render: {render_path}")
    
    # Alpha map
    alpha_path = output_dir / f'{prefix}_alpha.png'
    save_alpha_map(alpha_map, alpha_path)
    logger.info(f"Saved alpha map: {alpha_path}")
    
    # PSF profiles
    if not args.no_profiles and len(strokes) > 0:
        psf_path = output_dir / f'{prefix}_psf_profiles.png'
        plot_psf_profiles(renderer, strokes, args.psf_samples, psf_path)
        logger.info(f"Saved PSF profiles: {psf_path}")
    
    # Metadata (convert numpy types to Python native types for YAML)
    def convert_to_native(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(item) for item in obj)
        return obj
    
    metadata = {
        'num_strokes': len(strokes),
        'canvas_size_px': (canvas_h, canvas_w),
        'work_area_mm': (work_w, work_h),
        'render_time_s': float(render_time),
        'background': args.background,
        'lut_source': 'toy' if not args.lut_dir else str(args.lut_dir),
        'strokes': convert_to_native(strokes)
    }
    metadata_path = output_dir / f'{prefix}_metadata.yaml'
    fs.atomic_yaml_dump(metadata, metadata_path)
    logger.info(f"Saved metadata: {metadata_path}")
    
    logger.info("Preview complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
