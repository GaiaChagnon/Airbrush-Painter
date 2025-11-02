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
            work = info.data['work_area_mm']
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


# ============================================================================
# CPU RENDERER CONFIG (renderer_cpu.v1.yaml)
# ============================================================================

class VisibilityConfig(BaseModel):
    """Visibility thresholds for skipping imperceptible strokes."""
    min_alpha_visible: float = Field(default=0.0125, ge=0.0, le=1.0)
    min_delta_e_visible: float = Field(default=0.8, ge=0.0)
    min_stroke_coverage: float = Field(default=0.0001, ge=0.0, le=1.0)
    min_center_luminance_drop: float = Field(default=0.05, ge=0.0, le=1.0)


class WidthModel(BaseModel):
    """Width model: maps (z, v) → spray width in mm."""
    z_knots_mm: List[float] = Field(..., min_length=2)
    width_min_mm: List[float] = Field(..., min_length=2)
    width_max_mm: List[float] = Field(..., min_length=2)
    v_knots_mm_s: List[float] = Field(..., min_length=2)
    width_scale: List[float] = Field(..., min_length=2)
    
    @model_validator(mode='after')
    def validate_lengths(self) -> 'WidthModel':
        """Validate that all arrays have consistent lengths."""
        z_len = len(self.z_knots_mm)
        if len(self.width_min_mm) != z_len or len(self.width_max_mm) != z_len:
            raise ValueError("width_min_mm and width_max_mm must match z_knots_mm length")
        
        v_len = len(self.v_knots_mm_s)
        if len(self.width_scale) != v_len:
            raise ValueError("width_scale must match v_knots_mm_s length")
        
        return self


class ProfileConfig(BaseModel):
    """Radial opacity profile shape parameters."""
    type: str = Field(default="flat_core_gaussian_skirt")
    core_frac: float = Field(default=0.40, ge=0.0, le=1.0)
    skirt_sigma_frac: float = Field(default=0.28, ge=0.0, le=1.0)
    skirt_power: float = Field(default=1.8, ge=0.5, le=5.0)
    margin_factor: float = Field(default=1.5, ge=1.0, le=5.0)


class DepositionModel(BaseModel):
    """Deposition model: mass of opacity per unit length."""
    z_knots_mm: List[float] = Field(..., min_length=2)
    mass_per_sec: List[float] = Field(..., min_length=2)
    speed_exponent: float = Field(default=1.0, ge=0.0, le=2.0)
    k_mass: float = Field(default=2.5, ge=0.1, le=100.0)
    
    @model_validator(mode='after')
    def validate_lengths(self) -> 'DepositionModel':
        """Validate that arrays have consistent lengths."""
        if len(self.mass_per_sec) != len(self.z_knots_mm):
            raise ValueError("mass_per_sec must match z_knots_mm length")
        return self


class StampTrainConfig(BaseModel):
    """Stamp train mode configuration (alternative to distance transform)."""
    patterns: List[str] = Field(default_factory=list)
    unit_diam_mm: float = Field(default=1.0, ge=0.1, le=10.0)
    spacing_mm: float = Field(default=0.5, ge=0.05, le=5.0)
    jitter_mm: float = Field(default=0.08, ge=0.0, le=1.0)
    noise_gain: float = Field(default=0.12, ge=0.0, le=1.0)


class MixingConfig(BaseModel):
    """Color mixing/layering configuration."""
    mode: str = Field(default="layer_over")
    km_params_path: Optional[str] = None
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {'layer_over', 'kubelka_munk'}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v}")
        return v


class RandomnessConfig(BaseModel):
    """Randomness control for speckle/texture."""
    seed: int = Field(default=42, ge=0)
    speckle: bool = Field(default=True)
    speckle_gain: float = Field(default=0.08, ge=0.0, le=1.0)
    speckle_scale: float = Field(default=2.0, ge=0.5, le=10.0)


class SamplingConfig(BaseModel):
    """Sampling along polyline configuration."""
    max_step_mm: float = Field(default=0.25, ge=0.01, le=5.0)
    min_samples: int = Field(default=8, ge=2, le=1000)


class RendererCPUV1(BaseModel):
    """CPU renderer configuration (renderer_cpu.v1.yaml schema).
    
    Defines physical behavior for deterministic OpenCV-based airbrush simulation.
    Controls width, mass deposition, visibility gates, and profile shape.
    """
    schema: str = Field(default="renderer_cpu.v1")
    mode: str = Field(default="opencv_distance")
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    width_model: WidthModel
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    deposition: DepositionModel
    stamp_train: StampTrainConfig = Field(default_factory=StampTrainConfig)
    mixing: MixingConfig = Field(default_factory=MixingConfig)
    randomness: RandomnessConfig = Field(default_factory=RandomnessConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {'opencv_distance', 'stamp_train'}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v}")
        return v
    
    @field_validator('schema')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "renderer_cpu.v1":
            raise ValueError(f"schema must be 'renderer_cpu.v1', got {v}")
        return v


def load_renderer_cpu_config(path: Union[str, Path]) -> RendererCPUV1:
    """Load and validate CPU renderer config from YAML.
    
    Parameters
    ----------
    path : str or Path
        Path to renderer_cpu.v1.yaml file
    
    Returns
    -------
    RendererCPUV1
        Validated config model
    
    Raises
    ------
    ValueError
        If config is invalid
    FileNotFoundError
        If file does not exist
    """
    from . import fs
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CPU renderer config not found: {path}")
    
    cfg = fs.load_yaml(path)
    return RendererCPUV1(**cfg)


# ============================================================================
# PEN TOOL SCHEMA V1
# ============================================================================

class PenMacros(BaseModel):
    """Pen tool G-code macros."""
    pen_up: str = Field(..., description="Pen up macro filename")
    pen_down: str = Field(..., description="Pen down macro filename")


class PenAcceleration(BaseModel):
    """Pen tool acceleration limits."""
    xy_mm_s2: float = Field(..., gt=0, description="XY acceleration (mm/s²)")


class PenJoins(BaseModel):
    """Pen path rendering/offsetting style."""
    cap_style: str = Field(..., description="Cap style for path ends")
    join_style: str = Field(..., description="Join style for path corners")
    
    @field_validator('cap_style')
    @classmethod
    def validate_cap_style(cls, v: str) -> str:
        allowed = {"round", "square", "flat"}
        if v not in allowed:
            raise ValueError(f"cap_style must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('join_style')
    @classmethod
    def validate_join_style(cls, v: str) -> str:
        allowed = {"round", "miter", "bevel"}
        if v not in allowed:
            raise ValueError(f"join_style must be one of {allowed}, got '{v}'")
        return v


class PenToolV1(BaseModel):
    """Pen tool schema v1 (physical pen tool configuration)."""
    schema: str = Field(..., description="Schema version")
    name: str = Field(..., description="Tool name")
    tip_diameter_mm: float = Field(..., ge=0.1, le=2.0, description="Pen tip diameter (mm)")
    overlap_frac: float = Field(..., ge=0.0, le=0.5, description="Overlap fraction between passes")
    max_passes: int = Field(..., ge=1, le=10, description="Maximum overshading stacks")
    feed_mm_s: float = Field(..., ge=10.0, le=300.0, description="Drawing feed rate (mm/s)")
    travel_mm_s: float = Field(..., ge=50.0, le=400.0, description="Travel feed rate (mm/s)")
    safe_z_mm: float = Field(..., ge=0.0, le=30.0, description="Safe Z for travel (mm)")
    draw_z_mm: float = Field(..., ge=-2.0, le=10.0, description="Pen-down Z (mm)")
    plunge_mm_s: float = Field(..., ge=1.0, le=50.0, description="Z feed rate (mm/s)")
    offset_mm: List[float] = Field(..., description="Tool offset from airbrush [dx, dy, dz]")
    macros: PenMacros
    accel: PenAcceleration
    joins: PenJoins
    
    @field_validator('schema')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "pen_tool.v1":
            raise ValueError(f"Expected schema 'pen_tool.v1', got '{v}'")
        return v
    
    @field_validator('offset_mm')
    @classmethod
    def validate_offset(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError(f"offset_mm must have 3 elements [dx, dy, dz], got {len(v)}")
        for i, val in enumerate(v):
            if not (-50.0 <= val <= 50.0):
                raise ValueError(f"offset_mm[{i}] = {val} out of bounds [-50.0, 50.0]")
        return v


# ============================================================================
# PEN TRACER SCHEMA V2 (Edge + Gamut-Aware Hatching)
# ============================================================================

class PenTracerV2Output(BaseModel):
    """Output resolution settings."""
    target_height_px: Optional[int] = Field(None, description="Target height (width from aspect ratio), null for original")
    min_px: int = Field(512, ge=256, le=2048, description="Minimum resolution")
    max_px: int = Field(4096, ge=1024, le=8192, description="Maximum resolution")
    
    @field_validator('target_height_px')
    @classmethod
    def validate_target_height(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"target_height_px must be positive, got {v}")
        return v


class PenTracerV2EdgeDetection(BaseModel):
    """Edge detection configuration."""
    enabled: bool = Field(True, description="Enable edge extraction")
    # Bilateral filter preprocessing
    bilateral_d: int = Field(9, ge=5, le=15, description="Bilateral filter diameter")
    bilateral_sigma_color: float = Field(75.0, ge=50.0, le=150.0, description="Color smoothing")
    bilateral_sigma_space: float = Field(75.0, ge=50.0, le=150.0, description="Spatial smoothing")
    # Canny edge detection
    canny_low: float = Field(30.0, ge=10.0, le=100.0, description="Canny low threshold")
    canny_high: float = Field(100.0, ge=50.0, le=200.0, description="Canny high threshold")
    sigma_px: float = Field(1.2, ge=0.5, le=3.0, description="Gaussian blur sigma")
    # Morphological operations
    closing_kernel_size: int = Field(5, ge=3, le=9, description="Closing kernel size")
    merge_kernel_size: int = Field(3, ge=1, le=5, description="Merge kernel size")
    # Filtering & simplification
    min_length_px: int = Field(50, ge=10, le=100, description="Minimum edge length px")
    simplify_tol_px: float = Field(0.5, ge=0.1, le=5.0, description="Simplification tolerance")


class PenTracerV2CMYGamut(BaseModel):
    """CMY airbrush gamut definition."""
    min_luminance: float = Field(15.0, ge=0.0, le=50.0, description="Minimum L* achievable")
    max_chroma: float = Field(80.0, ge=0.0, le=150.0, description="Maximum chroma")
    hue_ranges: List[List[float]] = Field([[0, 360]], description="Hue ranges [start, end] degrees")


class PenTracerV2DarknessLevel(BaseModel):
    """Single darkness level for hatching (exclusive range)."""
    l_min: float = Field(..., ge=0.0, le=100.0, description="Minimum L* for this level (inclusive)")
    l_max: float = Field(..., ge=0.0, le=100.0, description="Maximum L* for this level (exclusive)")
    passes: int = Field(..., ge=1, le=5, description="Number of hatching passes")
    hatch_angles: List[float] = Field(..., description="Hatch angles in degrees")
    
    @field_validator('l_max')
    @classmethod
    def validate_range(cls, v: float, info) -> float:
        if 'l_min' in info.data and v <= info.data['l_min']:
            raise ValueError(f"l_max ({v}) must be greater than l_min ({info.data['l_min']})")
        return v


class PenTracerV2ShadowHatching(BaseModel):
    """Shadow hatching configuration."""
    enabled: bool = Field(True, description="Enable shadow hatching")
    gamut_aware: bool = Field(True, description="Only hatch out-of-gamut regions")
    cmy_gamut: PenTracerV2CMYGamut
    darkness_levels: List[PenTracerV2DarknessLevel]
    min_area_px: int = Field(500, ge=50, le=5000, description="Minimum shadow area")
    close_gaps_px: int = Field(0, ge=0, le=10, description="Morphological closing kernel size")
    spacing_scale: float = Field(1.0, ge=0.1, le=5.0, description="Hatch spacing multiplier")
    min_line_spacing_mm: float = Field(0.5, ge=0.1, le=5.0, description="Minimum line spacing")
    min_segment_length_mm: float = Field(0.0, ge=0.0, le=10.0, description="Minimum segment length to keep")
    max_hatch_coverage: float = Field(0.75, ge=0.0, le=1.0, description="Maximum hatch coverage fraction")


class PenTracerV2Calibration(BaseModel):
    """Calibration integration settings."""
    calibration_file: Optional[str] = Field(None, description="Path to calibration YAML")
    margin: float = Field(0.05, ge=0.0, le=0.3, description="Gamut expansion margin")


class PenTracerV2Debug(BaseModel):
    """Debug output settings."""
    save_intermediates: bool = Field(True, description="Save intermediate images")
    save_gamut_mask: bool = Field(True, description="Save out-of-gamut mask")
    save_edge_mask: bool = Field(True, description="Save edge mask")
    save_shadow_masks: bool = Field(True, description="Save shadow masks")


class PenTracerV2(BaseModel):
    """Pen tracer schema v2 (edge + gamut-aware hatching)."""
    schema: str = Field(..., description="Schema version")
    output: PenTracerV2Output
    edge_detection: PenTracerV2EdgeDetection
    shadow_hatching: PenTracerV2ShadowHatching
    calibration: PenTracerV2Calibration
    debug: PenTracerV2Debug
    
    @field_validator('schema')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "pen_tracer.v2":
            raise ValueError(f"Expected schema 'pen_tracer.v2', got '{v}'")
        return v


# ============================================================================
# PEN VECTORS SCHEMA V1
# ============================================================================

class PenPath(BaseModel):
    """Single pen path (polyline in image frame mm)."""
    id: str = Field(..., description="Unique path identifier")
    kind: str = Field(..., description="Path topology")
    role: str = Field(..., description="Path semantic role")
    tip_diameter_mm: float = Field(..., ge=0.1, le=2.0, description="Pen tip diameter (mm)")
    z_mm: float = Field(..., ge=-2.0, le=10.0, description="Pen Z position (mm)")
    feed_mm_s: float = Field(..., ge=10.0, le=300.0, description="XY feed rate (mm/s)")
    points_mm: List[List[float]] = Field(..., description="Polyline points [x, y] in mm")
    
    @field_validator('kind')
    @classmethod
    def validate_kind(cls, v: str) -> str:
        allowed = {"polyline", "polygon"}
        if v not in allowed:
            raise ValueError(f"kind must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"outline", "hatch", "fill"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('points_mm')
    @classmethod
    def validate_points(cls, v: List[List[float]]) -> List[List[float]]:
        for i, pt in enumerate(v):
            if len(pt) != 2:
                raise ValueError(f"Point {i} must have 2 coordinates, got {len(pt)}")
            x, y = pt
            if not (0.0 <= x <= 210.0):  # A4 width
                raise ValueError(f"Point {i} x={x} out of bounds [0, 210]")
            if not (0.0 <= y <= 297.0):  # A4 height
                raise ValueError(f"Point {i} y={y} out of bounds [0, 297]")
        return v


class PenVectorsMetadata(BaseModel):
    """Pen vectors metadata (provenance)."""
    tool_name: str = Field(..., description="Tool name")
    offset_mm: List[float] = Field(..., description="Tool offset [dx, dy, dz]")
    hatch_angles_deg: List[float] = Field(..., description="Hatch angles used")
    generated_at: str = Field(..., description="ISO 8601 timestamp")
    tracer_version: str = Field(..., description="Tracer schema version")


class PenVectorsV1(BaseModel):
    """Pen vectors schema v1 (serialization format for pen layer)."""
    schema: str = Field(..., description="Schema version")
    render_px: List[int] = Field(..., description="Rendering resolution [W, H]")
    work_area_mm: List[float] = Field(..., description="Canvas dimensions [W, H] mm")
    paths: List[PenPath] = Field(..., description="List of pen paths")
    metadata: PenVectorsMetadata
    
    @field_validator('schema')
    @classmethod
    def validate_schema(cls, v: str) -> str:
        if v != "pen_vectors.v1":
            raise ValueError(f"Expected schema 'pen_vectors.v1', got '{v}'")
        return v
    
    @field_validator('render_px')
    @classmethod
    def validate_render_px(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError(f"render_px must have 2 elements [W, H], got {len(v)}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"render_px dimensions must be positive, got {v}")
        return v
    
    @field_validator('work_area_mm')
    @classmethod
    def validate_work_area(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError(f"work_area_mm must have 2 elements [W, H], got {len(v)}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"work_area_mm dimensions must be positive, got {v}")
        return v


# ============================================================================
# PUBLIC API (PEN LAYER)
# ============================================================================

def load_pen_tool_config(path: Union[str, Path]) -> PenToolV1:
    """Load and validate pen tool config from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to pen_tool.v1.yaml file
    
    Returns
    -------
    PenToolV1
        Validated pen tool configuration
    
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
        raise FileNotFoundError(f"Pen tool config not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return PenToolV1(**data)
    except Exception as e:
        raise ValueError(f"Pen tool config validation failed at {path}: {e}") from e


def load_pen_tracer_v2_config(path: Union[str, Path]) -> PenTracerV2:
    """Load and validate pen tracer V2 config from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to pen_tracer.v2.yaml file
    
    Returns
    -------
    PenTracerV2
        Validated pen tracer V2 configuration
    
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
        raise FileNotFoundError(f"Pen tracer V2 config not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return PenTracerV2(**data)
    except Exception as e:
        raise ValueError(f"Pen tracer V2 config validation failed at {path}: {e}") from e


def load_pen_vectors(path: Union[str, Path]) -> PenVectorsV1:
    """Load and validate pen vectors from YAML.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to pen_vectors.v1.yaml file
    
    Returns
    -------
    PenVectorsV1
        Validated pen vectors
    
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
        raise FileNotFoundError(f"Pen vectors file not found: {path}")
    
    data = fs.load_yaml(path)
    try:
        return PenVectorsV1(**data)
    except Exception as e:
        raise ValueError(f"Pen vectors validation failed at {path}: {e}") from e
