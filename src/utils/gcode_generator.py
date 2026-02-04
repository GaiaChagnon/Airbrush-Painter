"""G-code generation from stroke and pen vectors.

Converts validated strokes (stroke.v1.yaml) and pen paths to GRBL-compatible G-code:
    - Coordinate frame transform: image (top-left, +Y down) → machine (bottom-left, +Y up)
    - Bézier linearization with interpolated Z and speed
    - Feed conversion: mm/s → F<mm/min>
    - Macro insertion: PURGE before each stroke, PEN_UP/PEN_DOWN for pen layer
    - Soft-limit validation against machine profile
    - Stroke ID comments: ; STROKE_ID: <id>

Outputs:
    <job>_cmy.gcode: Airbrush painting pass
    <job>_pen.gcode: Black pen layer
    <job>_manifest.yaml: Complete job provenance (job.v1.yaml schema)

GRBL-safe header (no RepRap codes):
    G21            ; mm units
    G90            ; absolute positioning
    G92 X0 Y0 Z0   ; set current position as origin (optional)

Machine-specific post-processors via strategy pattern (extensible).
Validates all moves against work_area_mm and feeds.max_xy_mm_s.

Usage:
    from src.utils import gcode_generator, validators, fs
    
    job_cfg = validators.load_job_config("job.yaml")
    machine_cfg = validators.load_machine_profile("machine.yaml")
    strokes = validators.validate_strokes_file("strokes.yaml")
    
    gcode_generator.generate_all_gcode(
        job_cfg, machine_cfg, strokes,
        output_dir="gcode_output/"
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import logging

from . import fs, geometry, compute, validators

logger = logging.getLogger(__name__)

# Track warned macros to avoid log spam
_warned_macros: Set[str] = set()


# ============================================================================
# COORDINATE FRAME TRANSFORMS
# ============================================================================

def image_mm_to_machine_mm(
    x_img_mm: torch.Tensor,
    canvas_bounds: Tuple[float, float, float, float],
    flip_y: bool = True
) -> torch.Tensor:
    """Transform image-frame mm to machine-frame mm with canvas offset.
    
    Parameters
    ----------
    x_img_mm : torch.Tensor
        Coordinates in image frame (relative to canvas), shape (..., 2) with (x, y)
        Image frame: origin top-left of canvas, +Y down
    canvas_bounds : Tuple[float, float, float, float]
        Canvas bounds (x_min, x_max, y_min, y_max) in machine coordinates
    flip_y : bool
        True if machine uses bottom-left origin (+Y up), default True
    
    Returns
    -------
    torch.Tensor
        Coordinates in machine frame (absolute), shape (..., 2) with (x, y)
        Machine frame: origin at machine (0,0), +Y up (standard G-code)
    
    Notes
    -----
    This transform is applied once at G-code generation boundary.
    Internal pipeline stays in image frame for consistency with rendering.
    Accounts for canvas offset from machine origin.
    """
    x_min, x_max, y_min, y_max = canvas_bounds
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    x_mach_mm = x_img_mm.clone()
    
    if flip_y:
        # Flip Y (image top→machine top becomes image bottom→machine bottom)
        x_mach_mm[..., 1] = canvas_height - x_img_mm[..., 1]
    
    # Add canvas offset to get absolute machine coordinates
    x_mach_mm[..., 0] = x_mach_mm[..., 0] + x_min
    x_mach_mm[..., 1] = x_mach_mm[..., 1] + y_min
    
    return x_mach_mm


# ============================================================================
# G-CODE GENERATION
# ============================================================================

def generate_gcode_header(machine_cfg: validators.MachineV1) -> List[str]:
    """Generate GRBL-safe G-code header.
    
    Parameters
    ----------
    machine_cfg : validators.MachineV1
        Machine configuration
    
    Returns
    -------
    List[str]
        Header lines (with newlines)
    """
    lines = [
        "; --- AIRBRUSH JOB START ---\n",
        f"; Machine: {machine_cfg.gcode_flavor}\n",
        f"; Work area: {machine_cfg.work_area_mm.x}x{machine_cfg.work_area_mm.y}x{machine_cfg.work_area_mm.z} mm\n",
        "G21            ; mm units\n",
        "G90            ; absolute positioning\n",
    ]
    
    # Optional: zero current position (machine-specific)
    if machine_cfg.gcode_flavor == "grbl_1.1f":
        lines.append("; G92 X0 Y0 Z0   ; zero current position (if needed)\n")
    
    lines.append("\n")
    return lines


def load_macro(machine_cfg: validators.MachineV1, macro_name: str) -> List[str]:
    """Load G-code macro from file.
    
    Parameters
    ----------
    machine_cfg : validators.MachineV1
        Machine configuration
    macro_name : str
        Macro name ('purge', 'pen_up', 'pen_down')
    
    Returns
    -------
    List[str]
        Macro lines (with newlines), or empty list if file not found
    
    Notes
    -----
    Looks for macro file in machine_cfg.macros.include_dir.
    Returns empty list if file doesn't exist (graceful degradation).
    Warnings are debounced to avoid log spam.
    """
    global _warned_macros
    
    macro_filename = getattr(machine_cfg.macros, macro_name)
    macro_path = Path(machine_cfg.macros.include_dir) / macro_filename
    
    if not macro_path.exists():
        # Warn once per unique macro path
        macro_key = str(macro_path)
        if macro_key not in _warned_macros:
            logger.warning(f"Macro file not found: {macro_path}, skipping")
            _warned_macros.add(macro_key)
        return []
    
    try:
        with open(macro_path, 'r') as f:
            lines = f.readlines()
        # Ensure newlines
        return [line if line.endswith('\n') else line + '\n' for line in lines]
    except Exception as e:
        logger.warning(f"Failed to load macro {macro_path}: {e}, skipping")
        return []


def validate_soft_limits(
    x: float, y: float, z: float,
    machine_cfg: validators.MachineV1,
    z_min: float = 0.0
) -> None:
    """Validate move against soft limits.
    
    Parameters
    ----------
    x, y, z : float
        Machine coordinates (mm)
    machine_cfg : validators.MachineV1
        Machine configuration
    z_min : float, optional
        Minimum allowed Z value (default 0.0). Use negative values for pen tools
        that can press below surface (e.g., -2.0 for pen drawing).
    
    Raises
    ------
    ValueError
        If move violates soft limits
    """
    if not machine_cfg.safety.soft_limits:
        return  # Soft limits disabled
    
    work = machine_cfg.work_area_mm
    if not (0 <= x <= work.x):
        raise ValueError(f"X={x:.2f} out of bounds [0, {work.x}]")
    if not (0 <= y <= work.y):
        raise ValueError(f"Y={y:.2f} out of bounds [0, {work.y}]")
    if not (z_min <= z <= work.z):
        raise ValueError(f"Z={z:.2f} out of bounds [{z_min}, {work.z}]")


def linearize_stroke(
    stroke: validators.StrokeV1,
    machine_cfg: validators.MachineV1,
    max_err_mm: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linearize Bézier stroke into polyline with interpolated Z and speed.
    
    Parameters
    ----------
    stroke : validators.StrokeV1
        Validated stroke (image frame, relative to canvas)
    machine_cfg : validators.MachineV1
        Machine configuration (includes canvas bounds)
    max_err_mm : float
        Maximum flattening error (mm), default 0.5
    
    Returns
    -------
    pts_mach : torch.Tensor
        Machine-frame XY points (absolute), shape (N, 2), dtype float32
    z_interp : torch.Tensor
        Interpolated Z values, shape (N,), dtype float32
    v_interp : torch.Tensor
        Interpolated speeds (mm/s), shape (N,), dtype float32
    
    Notes
    -----
    Transforms to machine frame: image (top-left of canvas) → machine (absolute).
    Accounts for canvas offset from machine origin.
    Validates all points against machine soft limits.
    """
    # Extract Bézier control points (image frame, relative to canvas)
    bez = stroke.bezier
    q = torch.tensor([
        [bez.p1[0], bez.p1[1]],
        [bez.p2[0], bez.p2[1]],
        [bez.p3[0], bez.p3[1]],
        [bez.p4[0], bez.p4[1]],
    ], dtype=torch.float32)
    
    # Flatten Bézier curve
    pts_img = geometry.bezier_cubic_polyline(
        q[0], q[1], q[2], q[3],
        max_err_mm=max_err_mm
    )
    
    # Transform to machine frame (absolute coordinates)
    canvas = machine_cfg.canvas_mm
    canvas_bounds = (canvas.x_min, canvas.x_max, canvas.y_min, canvas.y_max)
    pts_mach = image_mm_to_machine_mm(
        pts_img,
        canvas_bounds=canvas_bounds,
        flip_y=True
    )
    
    # Compute arc-length parametrization for interpolation
    # t_vals will be [0, ..., 1] based on cumulative arc length
    if len(pts_img) > 1:
        # Compute cumulative arc length
        diffs = pts_img[1:] - pts_img[:-1]
        segment_lengths = torch.norm(diffs, dim=1)
        cumulative_lengths = torch.cat([torch.tensor([0.0]), torch.cumsum(segment_lengths, dim=0)])
        total_length = cumulative_lengths[-1]
        if total_length > 0:
            t_vals = cumulative_lengths / total_length
        else:
            t_vals = torch.linspace(0, 1, len(pts_img))
    else:
        t_vals = torch.tensor([0.0])
    
    # Interpolate Z and speed along arc length
    z0, z1 = stroke.z_profile.z0, stroke.z_profile.z1
    v0, v1 = stroke.speed_profile.v0, stroke.speed_profile.v1
    
    z_interp = z0 + (z1 - z0) * t_vals
    v_interp = v0 + (v1 - v0) * t_vals
    
    # Validate soft limits
    if machine_cfg.safety.soft_limits:
        for i in range(len(pts_mach)):
            validate_soft_limits(
                pts_mach[i, 0].item(),
                pts_mach[i, 1].item(),
                z_interp[i].item(),
                machine_cfg
            )
    
    return pts_mach, z_interp, v_interp


def speed_mm_s_to_feed(
    v_mm_s: float,
    feed_units: str,
    machine_cfg: Optional[validators.MachineV1] = None
) -> float:
    """Convert speed (mm/s) to G-code feed rate with clamping.
    
    Parameters
    ----------
    v_mm_s : float
        Speed in mm/s
    feed_units : str
        Feed units ('mm/min' or 'mm/s')
    machine_cfg : Optional[validators.MachineV1]
        Machine config for clamping, None to skip clamping
    
    Returns
    -------
    float
        Feed rate in specified units, clamped to safe range
    
    Notes
    -----
    Always enforces F >= 1.0 to avoid F0 (which confuses some controllers).
    If machine_cfg provided, clamps to max_xy_mm_s.
    """
    # Convert to target units
    if feed_units == "mm/min":
        feed = v_mm_s * 60.0
    elif feed_units == "mm/s":
        feed = v_mm_s
    else:
        raise ValueError(f"Unknown feed units: {feed_units}")
    
    # Enforce minimum (never F0)
    feed = max(1.0, feed)
    
    # Clamp to machine limits if provided
    if machine_cfg:
        max_feed = machine_cfg.feeds.max_xy_mm_s * (60.0 if feed_units == "mm/min" else 1.0)
        feed = min(feed, max_feed)
    
    return feed


def generate_stroke_gcode(
    stroke: validators.StrokeV1,
    machine_cfg: validators.MachineV1,
    include_purge: bool = True
) -> List[str]:
    """Generate G-code for a single stroke.
    
    Parameters
    ----------
    stroke : validators.StrokeV1
        Validated stroke (image frame, relative to canvas)
    machine_cfg : validators.MachineV1
        Machine configuration (includes canvas bounds)
    include_purge : bool
        Include PURGE macro before stroke, default True
    
    Returns
    -------
    List[str]
        G-code lines (with newlines)
    
    Raises
    ------
    ValueError
        If stroke violates machine soft limits
    """
    lines = []
    
    # Stroke ID comment
    lines.append(f"; STROKE_ID: {stroke.id}\n")
    
    # Optional PURGE macro
    if include_purge:
        purge_lines = load_macro(machine_cfg, "purge")
        if purge_lines:
            lines.extend(purge_lines)
    
    # Linearize stroke (transforms to absolute machine coordinates)
    pts_mach, z_interp, v_interp = linearize_stroke(
        stroke, machine_cfg
    )
    
    # Clamp speeds to machine limits
    v_interp = torch.clamp(v_interp, 1.0, machine_cfg.feeds.max_xy_mm_s)
    z_interp = torch.clamp(z_interp, 0.0, machine_cfg.work_area_mm.z)
    
    # Safe travel Z (80% of max Z, or use safety.travel_z_mm if present)
    safe_z = getattr(machine_cfg.safety, "travel_z_mm", machine_cfg.work_area_mm.z * 0.8)
    
    # Rapid to start position at safe Z
    x0 = pts_mach[0, 0].item()
    y0 = pts_mach[0, 1].item()
    validate_soft_limits(x0, y0, safe_z, machine_cfg)
    lines.append(f"G0 X{x0:.3f} Y{y0:.3f} Z{safe_z:.3f}\n")
    
    # Optional spray on macro (graceful if not defined)
    if hasattr(machine_cfg.macros, "spray_on"):
        spray_on_lines = load_macro(machine_cfg, "spray_on")
        if spray_on_lines:
            lines.extend(spray_on_lines)
    
    # Generate painting moves
    for i in range(len(pts_mach)):
        x = pts_mach[i, 0].item()
        y = pts_mach[i, 1].item()
        z = z_interp[i].item()
        v = v_interp[i].item()
        
        feed = speed_mm_s_to_feed(v, machine_cfg.feed_units, machine_cfg)
        
        # G1: Linear move with feed rate
        lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}\n")
    
    # Optional spray off macro (graceful if not defined)
    if hasattr(machine_cfg.macros, "spray_off"):
        spray_off_lines = load_macro(machine_cfg, "spray_off")
        if spray_off_lines:
            lines.extend(spray_off_lines)
    
    lines.append("\n")
    return lines


def generate_cmy_gcode(
    strokes: validators.StrokesFileV1,
    machine_cfg: validators.MachineV1,
    output_path: Union[str, Path]
) -> None:
    """Generate G-code for CMY airbrush pass.
    
    Parameters
    ----------
    strokes : validators.StrokesFileV1
        Validated strokes container
    machine_cfg : validators.MachineV1
        Machine configuration (includes canvas bounds)
    output_path : Union[str, Path]
        Output G-code file path
    
    Notes
    -----
    Writes file atomically via fs.atomic_write_bytes().
    Canvas bounds implicit from machine_cfg.
    """
    logger.info(f"Generating CMY G-code with {len(strokes.strokes)} strokes...")
    
    lines = []
    
    # Header
    lines.extend(generate_gcode_header(machine_cfg))
    
    # Strokes
    for i, stroke in enumerate(strokes.strokes):
        try:
            stroke_lines = generate_stroke_gcode(
                stroke, machine_cfg,
                include_purge=True
            )
            lines.extend(stroke_lines)
        except Exception as e:
            logger.error(f"Failed to generate G-code for stroke {i} (ID: {stroke.id}): {e}")
            raise ValueError(f"Stroke {i} G-code generation failed: {e}") from e
    
    # Footer
    lines.append("; --- AIRBRUSH JOB END ---\n")
    
    # Write atomically
    gcode_bytes = ''.join(lines).encode('utf-8')
    fs.atomic_write_bytes(Path(output_path), gcode_bytes)
    
    logger.info(f"CMY G-code written to {output_path}")


def generate_pen_gcode(
    pen_vectors: validators.PenVectorsV1,
    machine_cfg: validators.MachineV1,
    pen_tool_cfg: validators.PenToolV1,
    output_path: Union[str, Path]
) -> None:
    """Generate G-code for pen layer with tool offset.
    
    Parameters
    ----------
    pen_vectors : validators.PenVectorsV1
        Validated pen vectors (image frame, relative to canvas)
    machine_cfg : validators.MachineV1
        Machine configuration (includes canvas bounds)
    pen_tool_cfg : validators.PenToolV1
        Pen tool configuration (feeds, Z, offset)
    output_path : Union[str, Path]
        Output G-code file path
    
    Notes
    -----
    Writes file atomically via fs.atomic_write_bytes().
    Uses PEN_UP/PEN_DOWN macros for pen control.
    Applies tool offset (dx, dy, dz) after machine frame transform.
    Canvas bounds implicit from machine_cfg.
    """
    logger.info(f"Generating pen G-code with {len(pen_vectors.paths)} paths...")
    
    lines = []
    
    # Header
    lines.extend(generate_gcode_header(machine_cfg))
    lines.append(f"; Tool: {pen_tool_cfg.name}\n")
    lines.append(f"; Tool offset: {pen_tool_cfg.offset_mm} mm\n")
    lines.append("\n")
    
    # Pen up initially
    pen_up_lines = load_macro(machine_cfg, "pen_up")
    lines.extend(pen_up_lines)
    
    # Tool offset
    tool_offset = torch.tensor(pen_tool_cfg.offset_mm[:2], dtype=torch.float32)  # [dx, dy]
    tool_offset_z = pen_tool_cfg.offset_mm[2]
    
    # Canvas bounds for coordinate transform
    canvas = machine_cfg.canvas_mm
    canvas_bounds = (canvas.x_min, canvas.x_max, canvas.y_min, canvas.y_max)
    
    # Process paths
    for path_obj in pen_vectors.paths:
        lines.append(f"; PATH_ID: {path_obj.id}\n")
        lines.append(f"; Role: {path_obj.role}, Kind: {path_obj.kind}\n")
        
        polyline_mm = path_obj.points_mm
        if len(polyline_mm) == 0:
            continue
        
        # Convert to tensor and transform to machine frame (absolute coordinates)
        pts_img = torch.tensor(polyline_mm, dtype=torch.float32)
        pts_mach = image_mm_to_machine_mm(pts_img, canvas_bounds, flip_y=True)
        
        # Apply tool offset
        pts_mach_offset = pts_mach + tool_offset.unsqueeze(0)
        
        # Z with offset
        pen_z = path_obj.z_mm + tool_offset_z
        
        # Feed rate
        pen_feed = speed_mm_s_to_feed(
            path_obj.feed_mm_s,
            machine_cfg.feed_units,
            machine_cfg
        )
        
        # Move to start (pen up)
        x0, y0 = pts_mach_offset[0, 0].item(), pts_mach_offset[0, 1].item()
        
        # Safe travel Z
        travel_z = pen_tool_cfg.safe_z_mm + tool_offset_z
        validate_soft_limits(x0, y0, travel_z, machine_cfg, z_min=-2.0)
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} Z{travel_z:.3f}\n")
        
        # Pen down (plunge to drawing Z)
        pen_down_lines = load_macro(machine_cfg, "pen_down")
        lines.extend(pen_down_lines)
        
        # Plunge feed
        plunge_feed = speed_mm_s_to_feed(
            pen_tool_cfg.plunge_mm_s,
            machine_cfg.feed_units,
            machine_cfg
        )
        validate_soft_limits(x0, y0, pen_z, machine_cfg, z_min=-2.0)
        lines.append(f"G1 Z{pen_z:.3f} F{plunge_feed:.1f}\n")
        
        # Draw polyline
        for i in range(1, len(pts_mach_offset)):
            x = pts_mach_offset[i, 0].item()
            y = pts_mach_offset[i, 1].item()
            validate_soft_limits(x, y, pen_z, machine_cfg, z_min=-2.0)
            lines.append(f"G1 X{x:.3f} Y{y:.3f} F{pen_feed:.1f}\n")
        
        # Pen up
        lines.append(f"G0 Z{travel_z:.3f}\n")
        lines.extend(pen_up_lines)
        lines.append("\n")
    
    # Footer
    lines.append("; --- PEN JOB END ---\n")
    
    # Write atomically
    gcode_bytes = ''.join(lines).encode('utf-8')
    fs.atomic_write_bytes(Path(output_path), gcode_bytes)
    
    logger.info(f"Pen G-code written to {output_path}")


def generate_all_gcode(
    job_cfg: validators.JobV1,
    machine_cfg: validators.MachineV1,
    strokes: validators.StrokesFileV1,
    pen_vectors: Optional[validators.PenVectorsV1] = None,
    pen_tool_cfg: Optional[validators.PenToolV1] = None,
    output_dir: Union[str, Path] = "gcode_output"
) -> Dict[str, Path]:
    """Generate all G-code files for a job.
    
    Parameters
    ----------
    job_cfg : validators.JobV1
        Validated job configuration
    machine_cfg : validators.MachineV1
        Validated machine configuration
    strokes : validators.StrokesFileV1
        Validated strokes container
    pen_vectors : Optional[validators.PenVectorsV1]
        Optional validated pen vectors (image frame)
    pen_tool_cfg : Optional[validators.PenToolV1]
        Optional pen tool configuration (required if pen_vectors provided)
    output_dir : Union[str, Path]
        Output directory, default "gcode_output"
    
    Returns
    -------
    Dict[str, Path]
        Mapping of artifact names to output paths
        Keys: 'cmy_gcode', 'pen_gcode', 'manifest'
    
    Raises
    ------
    ValueError
        If pen_vectors provided but pen_tool_cfg is None
    
    Notes
    -----
    Creates output directory if needed.
    Writes all files atomically.
    Generates manifest.yaml with full job provenance.
    """
    global _warned_macros
    _warned_macros.clear()  # Reset per-job to allow warnings in new runs
    
    output_dir = Path(output_dir)
    fs.ensure_dir(output_dir)
    
    # Generate CMY G-code (platform-safe basename)
    # Canvas bounds implicit from machine_cfg
    cmy_gcode_path = output_dir / Path(job_cfg.artifacts.cmy_gcode_out).name
    generate_cmy_gcode(strokes, machine_cfg, cmy_gcode_path)
    
    # Generate pen G-code (if provided)
    pen_gcode_path = None
    if pen_vectors is not None and job_cfg.artifacts.pen_gcode_out:
        if pen_tool_cfg is None:
            raise ValueError("pen_tool_cfg required when pen_vectors provided")
        
        pen_gcode_path = output_dir / Path(job_cfg.artifacts.pen_gcode_out).name
        generate_pen_gcode(pen_vectors, machine_cfg, pen_tool_cfg, pen_gcode_path)
    
    # Generate manifest (platform-safe basename)
    manifest_path = output_dir / Path(job_cfg.artifacts.manifest_out).name
    manifest_data = {
        'schema': 'job.v1',
        'machine_profile': job_cfg.machine_profile,
        'inputs': job_cfg.inputs.dict(),
        'limits': job_cfg.limits.dict(),
        'passes': [p.dict() for p in job_cfg.passes],
        'artifacts': {
            'cmy_gcode': str(cmy_gcode_path),
            'pen_gcode': str(pen_gcode_path) if pen_gcode_path else None,
            'manifest': str(manifest_path),
        }
    }
    fs.atomic_yaml_dump(manifest_data, manifest_path)
    
    logger.info(f"Job manifest written to {manifest_path}")
    
    return {
        'cmy_gcode': cmy_gcode_path,
        'pen_gcode': pen_gcode_path,
        'manifest': manifest_path,
    }
