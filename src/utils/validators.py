"""YAML schema validation and config loading.

Provides centralized validation for all configuration files using pydantic:
    - Stroke schema (stroke.v1.yaml): Bézier parameters, z/speed profiles, CMY color
    - LUT schema (luts.v1.yaml): Color/PSF/alpha lookup table metadata
    - Machine schema (machine.v1.yaml): Work area, feeds, G-code flavor, macros
    - Job schema (job.v1.yaml): Complete job specification for inference

All modules must use these validators to load configs for fail-fast error detection
with actionable messages (line numbers, offending keys, expected ranges).

Units:
    - Geometry: millimeters (mm)
    - Speed: mm/s
    - Color: [0.0, 1.0] for CMY and RGB

Usage:
    from src.utils import validators, fs
    
    job_cfg = validators.load_job_config("job.yaml")
    machine_cfg = validators.load_machine_profile("machine.yaml")
    strokes = validators.validate_strokes_file("strokes.yaml")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# STROKE SCHEMA V1
# ============================================================================

class BezierControlPoints(BaseModel):
    """Cubic Bézier control points (4 points, xy in mm)."""
    p1: Tuple[float, float] = Field(..., description="Start point (x, y) in mm")
    p2: Tuple[float, float] = Field(..., description="First control point (x, y) in mm")
    p3: Tuple[float, float] = Field(..., description="Second control point (x, y) in mm")
    p4: Tuple[float, float] = Field(..., description="End point (x, y) in mm")


class ZProfile(BaseModel):
    """Nozzle height at start and end (mm)."""
    z0: float = Field(..., ge=0.0, le=30.0, description="Height at start (mm)")
    z1: float = Field(..., ge=0.0, le=30.0, description="Height at end (mm)")


class SpeedProfile(BaseModel):
    """Speed at start and end (mm/s)."""
    v0: float = Field(..., ge=1.0, le=300.0, description="Speed at start (mm/s)")
    v1: float = Field(..., ge=1.0, le=300.0, description="Speed at end (mm/s)")


class ColorCMY(BaseModel):
    """CMY color (0.0-1.0)."""
    c: float = Field(..., ge=0.0, le=1.0, description="Cyan component")
    m: float = Field(..., ge=0.0, le=1.0, description="Magenta component")
    y: float = Field(..., ge=0.0, le=1.0, description="Yellow component")


class StrokeV1(BaseModel):
    """Single stroke definition (stroke.v1.yaml schema).
    
    All geometry in millimeters, image frame (top-left origin, +Y down).
    """
    id: str = Field(..., description="Unique stroke identifier (epoch-idx-uuid)")
    bezier: BezierControlPoints
    z_profile: ZProfile
    speed_profile: SpeedProfile
    color_cmy: ColorCMY
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or len(v) < 5:
            raise ValueError(f"Stroke ID must be non-empty and at least 5 chars, got: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_geometry_bounds(self) -> 'StrokeV1':
        """Validate control points are within A4 bounds."""
        bounds_x = (0.0, 210.0)
        bounds_y = (0.0, 297.0)
        
        bezier = self.bezier
        if bezier:
            for i, pt_name in enumerate(['p1', 'p2', 'p3', 'p4'], 1):
                pt = getattr(bezier, pt_name)
                x, y = pt
                if not (bounds_x[0] <= x <= bounds_x[1]):
                    raise ValueError(
                        f"Control point {pt_name} x={x:.2f} out of bounds [{bounds_x[0]}, {bounds_x[1]}]"
                    )
                if not (bounds_y[0] <= y <= bounds_y[1]):
                    raise ValueError(
                        f"Control point {pt_name} y={y:.2f} out of bounds [{bounds_y[0]}, {bounds_y[1]}]"
                    )
        return self


class StrokesFileV1(BaseModel):
    """Container for multiple strokes (YAML file format)."""
    schema_version: str = Field("stroke.v1", alias="schema", description="Schema version")
    strokes: List[StrokeV1] = Field(..., description="List of strokes")
    
    class Config:
        allow_population_by_field_name = True
    
    @field_validator('schema_version')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "stroke.v1":
            raise ValueError(f"Expected schema 'stroke.v1', got '{v}'")
        return v


# ============================================================================
# LUT SCHEMA V1
# ============================================================================

class ColorLUTGrid(BaseModel):
    """Color LUT grid dimensions."""
    c: int = Field(11, ge=2, description="Number of C samples")
    m: int = Field(11, ge=2, description="Number of M samples")
    y: int = Field(11, ge=2, description="Number of Y samples")


class ColorLUT(BaseModel):
    """Color LUT (CMY → linear RGB)."""
    grid: ColorLUTGrid
    domain: Tuple[float, float] = Field((0.0, 1.0), description="CMY input range")
    shape: Tuple[int, int, int, int] = Field(..., description="[Nc x Nm x Ny x 3]")
    dtype: str = Field("float32", description="Tensor dtype")
    values_path: str = Field(..., description="Path to .pt file")
    description: str = Field("Trilinear interpolation on CMY cube → linear RGB [0,1]")
    
    @model_validator(mode='after')
    def validate_shape_consistency(self) -> 'ColorLUT':
        """Ensure shape matches grid."""
        grid = self.grid
        shape = self.shape
        if grid and shape:
            expected = (grid.c, grid.m, grid.y, 3)
            if shape != expected:
                raise ValueError(f"Shape {shape} doesn't match grid {expected}")
        return self


class AlphaLUTAxes(BaseModel):
    """Alpha LUT axes."""
    z: List[float] = Field(..., description="Z samples (mm)")
    v: List[float] = Field(..., description="Speed samples (mm/s)")


class AlphaLUT(BaseModel):
    """Alpha LUT (Z, speed → coverage factor)."""
    axes: AlphaLUTAxes
    shape: Tuple[int, int] = Field(..., description="[Nz x Nv]")
    dtype: str = Field("float32", description="Tensor dtype")
    values_path: str = Field(..., description="Path to .pt file")
    description: str = Field("Bilinear interpolation on Z×V grid → coverage [0,1]")
    
    @model_validator(mode='after')
    def validate_shape_consistency(self) -> 'AlphaLUT':
        """Ensure shape matches axes."""
        axes = self.axes
        shape = self.shape
        if axes and shape:
            expected = (len(axes.z), len(axes.v))
            if shape != expected:
                raise ValueError(f"Shape {shape} doesn't match axes {expected}")
        return self


class PSFLUTAxes(BaseModel):
    """PSF LUT axes."""
    z: List[float] = Field(..., description="Z samples (mm)")
    v: List[float] = Field(..., description="Speed samples (mm/s)")


class PSFLUT(BaseModel):
    """PSF LUT (Z, speed → point spread function)."""
    axes: PSFLUTAxes
    kernel_size: int = Field(15, ge=3, description="K×K kernel (must be odd)")
    shape: Tuple[int, int, int, int] = Field(..., description="[Nz x Nv x K x K]")
    dtype: str = Field("float32", description="Tensor dtype")
    values_path: str = Field(..., description="Path to .pt file")
    description: str = Field("Bilinear interpolation on Z×V grid → normalized Gaussian kernel")
    
    @field_validator('kernel_size')
    @classmethod
    def validate_odd_kernel(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_shape_consistency(self) -> 'PSFLUT':
        """Ensure shape matches axes and kernel."""
        axes = self.axes
        shape = self.shape
        kernel_size = self.kernel_size
        if axes and shape and kernel_size:
            expected = (len(axes.z), len(axes.v), kernel_size, kernel_size)
            if shape != expected:
                raise ValueError(f"Shape {shape} doesn't match expected {expected}")
        return self


class LUTMetadata(BaseModel):
    """LUT file metadata."""
    machine_profile: str = Field(..., description="Machine profile identifier")
    calibration_date: str = Field(..., description="Calibration date (YYYY-MM-DD)")
    operator: str = Field(..., description="Operator name")


class LUTsV1(BaseModel):
    """LUT schema v1 (complete config file)."""
    schema_version: str = Field("luts.v1", alias="schema", description="Schema version")
    semver: str = Field("1.0.0", description="Semantic version")
    color_lut: ColorLUT
    alpha_lut: AlphaLUT
    psf_lut: PSFLUT
    metadata: LUTMetadata
    
    class Config:
        allow_population_by_field_name = True
    
    @field_validator('schema_version')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "luts.v1":
            raise ValueError(f"Expected schema 'luts.v1', got '{v}'")
        return v


# ============================================================================
# MACHINE SCHEMA V1
# ============================================================================

class WorkArea(BaseModel):
    """Machine physical work area dimensions (mm), origin at (0,0)."""
    x: float = Field(..., gt=0, description="Max X travel (mm)")
    y: float = Field(..., gt=0, description="Max Y travel (mm)")
    z: float = Field(..., gt=0, description="Max Z travel (mm)")


class CanvasBounds(BaseModel):
    """Canvas/painting surface bounds (mm) within machine work area."""
    x_min: float = Field(..., ge=0, description="Canvas X minimum (mm)")
    x_max: float = Field(..., gt=0, description="Canvas X maximum (mm)")
    y_min: float = Field(..., ge=0, description="Canvas Y minimum (mm)")
    y_max: float = Field(..., gt=0, description="Canvas Y maximum (mm)")
    
    @field_validator('x_max')
    @classmethod
    def validate_x_range(cls, v, info):
        if info.data.get('x_min') and v <= info.data['x_min']:
            raise ValueError(f"x_max ({v}) must be > x_min ({values['x_min']})")
        return v
    
    @field_validator('y_max')
    @classmethod
    def validate_y_range(cls, v, info):
        if info.data.get('y_min') and v <= info.data['y_min']:
            raise ValueError(f"y_max ({v}) must be > y_min ({values['y_min']})")
        return v


class Feeds(BaseModel):
    """Feed rate limits (mm/s)."""
    max_xy_mm_s: float = Field(..., gt=0, description="Max XY speed (mm/s)")
    max_z_mm_s: float = Field(..., gt=0, description="Max Z speed (mm/s)")
    rapid_mm_s: float = Field(..., gt=0, description="Rapid (G0) speed (mm/s)")


class Acceleration(BaseModel):
    """Acceleration limits (mm/s²)."""
    max_xy_mm_s2: float = Field(..., gt=0, description="Max XY acceleration (mm/s²)")
    max_z_mm_s2: float = Field(..., gt=0, description="Max Z acceleration (mm/s²)")


class Macros(BaseModel):
    """G-code macro file references."""
    include_dir: str = Field(..., description="Macro directory path")
    purge: str = Field(..., description="Purge macro filename")
    pen_up: str = Field(..., description="Pen up macro filename")
    pen_down: str = Field(..., description="Pen down macro filename")


class PurgeZone(BaseModel):
    """Safe purge area (off canvas), mm."""
    x: Tuple[float, float] = Field(..., description="X range [min, max]")
    y: Tuple[float, float] = Field(..., description="Y range [min, max]")
    z: Tuple[float, float] = Field(..., description="Z range [min, max]")


class Safety(BaseModel):
    """Safety settings."""
    soft_limits: bool = Field(True, description="Enable soft limit checking")
    purge_zone_mm: PurgeZone


class MachineV1(BaseModel):
    """Machine profile schema v1."""
    schema_version: str = Field("machine.v1", alias="schema", description="Schema version")
    work_area_mm: WorkArea
    canvas_mm: CanvasBounds
    gcode_flavor: str = Field(..., description="G-code flavor (grbl_1.1f, marlin_2.0, etc.)")
    units: str = Field("mm", description="Units (mm or inch)")
    feed_units: str = Field("mm/min", description="Feed units (mm/min or mm/s)")
    feeds: Feeds
    acceleration: Acceleration
    macros: Macros
    safety: Safety
    
    class Config:
        allow_population_by_field_name = True
    
    @field_validator('schema_version')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "machine.v1":
            raise ValueError(f"Expected schema 'machine.v1', got '{v}'")
        return v
    
    @field_validator('canvas_mm')
    @classmethod
    def validate_canvas_within_work_area(cls, v, info):
        if 'work_area_mm' in info.data:
            work = values['work_area_mm']
            if v.x_max > work.x:
                raise ValueError(f"Canvas x_max ({v.x_max}) exceeds machine work area ({work.x})")
            if v.y_max > work.y:
                raise ValueError(f"Canvas y_max ({v.y_max}) exceeds machine work area ({work.y})")
        return v
    
    @field_validator('gcode_flavor')
    @classmethod
    def validate_flavor(cls, v: str) -> str:
        allowed = ["grbl_1.1f", "marlin_2.0", "reprap"]
        if v not in allowed:
            raise ValueError(f"G-code flavor must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('units')
    @classmethod
    def validate_units(cls, v: str) -> str:
        if v not in ["mm", "inch"]:
            raise ValueError(f"Units must be 'mm' or 'inch', got '{v}'")
        return v
    
    @field_validator('feed_units')
    @classmethod
    def validate_feed_units(cls, v: str) -> str:
        if v not in ["mm/min", "mm/s"]:
            raise ValueError(f"Feed units must be 'mm/min' or 'mm/s', got '{v}'")
        return v


# ============================================================================
# JOB SCHEMA V1
# ============================================================================

class JobInputs(BaseModel):
    """Job input paths."""
    target_image_path: str = Field(..., description="Path to target image")
    pen_layer_path: Optional[str] = Field(None, description="Path to pen layer YAML (optional)")


class BBoxMM(BaseModel):
    """Bounding box in mm."""
    x: Tuple[float, float] = Field(..., description="X range [min, max]")
    y: Tuple[float, float] = Field(..., description="Y range [min, max]")
    z: Tuple[float, float] = Field(..., description="Z range [min, max]")


class JobLimits(BaseModel):
    """Job execution limits."""
    max_strokes: int = Field(..., gt=0, description="Maximum stroke count")
    bbox_mm: BBoxMM


class JobPass(BaseModel):
    """Single painting pass (cmy or pen)."""
    name: str = Field(..., description="Pass name (cmy or pen)")
    strokes_path: Optional[str] = Field(None, description="Path to strokes YAML")
    vectors_path: Optional[str] = Field(None, description="Path to pen vectors YAML")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v not in ["cmy", "pen"]:
            raise ValueError(f"Pass name must be 'cmy' or 'pen', got '{v}'")
        return v


class JobArtifacts(BaseModel):
    """Output artifact paths."""
    cmy_gcode_out: str = Field(..., description="CMY G-code output path")
    pen_gcode_out: Optional[str] = Field(None, description="Pen G-code output path")
    manifest_out: str = Field(..., description="Job manifest output path")


class JobV1(BaseModel):
    """Job schema v1 (complete job specification)."""
    schema_version: str = Field("job.v1", alias="schema", description="Schema version")
    machine_profile: str = Field(..., description="Machine profile identifier")
    inputs: JobInputs
    limits: JobLimits
    passes: List[JobPass] = Field(..., min_items=1, description="Ordered list of passes")
    artifacts: JobArtifacts
    
    class Config:
        allow_population_by_field_name = True
    
    @field_validator('schema_version')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "job.v1":
            raise ValueError(f"Expected schema 'job.v1', got '{v}'")
        return v


# ============================================================================
# PUBLIC API
# ============================================================================

def load_job_config(path: Union[str, Path]) -> JobV1:
    """Load and validate job config from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to job.v1.yaml file
    
    Returns
    -------
    JobV1
        Validated job configuration
    
    Raises
    ------
    FileNotFoundError
        If path doesn't exist
    ValueError
        If validation fails (with actionable error message)
    """
    from . import fs
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job config not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return JobV1(**data)
    except Exception as e:
        raise ValueError(f"Job config validation failed at {path}: {e}") from e


def load_machine_profile(path: Union[str, Path]) -> MachineV1:
    """Load and validate machine profile from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to machine.v1.yaml file
    
    Returns
    -------
    MachineV1
        Validated machine configuration
    
    Raises
    ------
    FileNotFoundError
        If path doesn't exist
    ValueError
        If validation fails (with actionable error message)
    """
    from . import fs
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Machine profile not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return MachineV1(**data)
    except Exception as e:
        raise ValueError(f"Machine profile validation failed at {path}: {e}") from e


def load_luts_config(path: Union[str, Path]) -> LUTsV1:
    """Load and validate LUT config from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to luts.v1.yaml file
    
    Returns
    -------
    LUTsV1
        Validated LUT configuration
    
    Raises
    ------
    FileNotFoundError
        If path doesn't exist
    ValueError
        If validation fails (with actionable error message)
    """
    from . import fs
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LUTs config not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return LUTsV1(**data)
    except Exception as e:
        raise ValueError(f"LUTs config validation failed at {path}: {e}") from e


def validate_strokes_file(path: Union[str, Path]) -> StrokesFileV1:
    """Load and validate strokes file from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to strokes YAML file
    
    Returns
    -------
    StrokesFileV1
        Validated strokes container
    
    Raises
    ------
    FileNotFoundError
        If path doesn't exist
    ValueError
        If validation fails (with actionable error message including stroke index)
    """
    from . import fs
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strokes file not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return StrokesFileV1(**data)
    except Exception as e:
        raise ValueError(f"Strokes file validation failed at {path}: {e}") from e


def get_stroke_bounds(machine_cfg_path: Union[str, Path] = "configs/machine_grbl_airbrush_v1.yaml") -> Dict[str, Tuple[float, float]]:
    """Get stroke parameter bounds from machine profile.
    
    Parameters
    ----------
    machine_cfg_path : Union[str, Path]
        Path to machine profile, default "configs/machine_grbl_airbrush_v1.yaml"
    
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Bounds dictionary: {param: (min, max)}
        Keys: 'x', 'y', 'z', 'speed', 'cmy'
    
    Notes
    -----
    Used by env and renderer for dynamic action scaling.
    Returns CANVAS bounds for XY (painting surface), machine Z limits.
    Single source of truth derived from machine profile.
    """
    machine_cfg = load_machine_profile(machine_cfg_path)
    canvas = machine_cfg.canvas_mm
    
    return {
        'x': (canvas.x_min, canvas.x_max),
        'y': (canvas.y_min, canvas.y_max),
        'z': (0.0, machine_cfg.work_area_mm.z),
        'speed': (1.0, machine_cfg.feeds.max_xy_mm_s),  # Min hardcoded at 1.0 mm/s
        'cmy': (0.0, 1.0),  # Color components
    }


def flatten_config_for_mlflow(cfg: Union[Dict, BaseModel]) -> Dict[str, Any]:
    """Flatten nested config dict for MLflow parameter logging.
    
    Parameters
    ----------
    cfg : Union[Dict, BaseModel]
        Nested config dict or pydantic model
    
    Returns
    -------
    Dict[str, Any]
        Flattened dict with dot-separated keys (e.g., 'agent.learning_rate')
    
    Notes
    -----
    MLflow params must be flat key-value pairs.
    Lists/dicts are converted to JSON strings if needed.
    """
    if isinstance(cfg, BaseModel):
        cfg = cfg.dict()
    
    def _flatten(d: Dict, parent_key: str = '') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to JSON strings for MLflow
                import json
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    return _flatten(cfg)
