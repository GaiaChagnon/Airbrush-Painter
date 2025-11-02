"""Parameter dataclasses for pen tracer optimization.

Defines:
- EdgeParams: Edge detection and vectorization parameters (15 params)
- HatchParams: Shadow hatching parameters (6 params)
- GlobalParams: Fixed pen tool and gamut specifications
- Optimization bounds for all tunable parameters
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, List, Any
import yaml


@dataclass
class EdgeParams:
    """Edge detection and vectorization parameters.
    
    These control how outlines are extracted from images.
    All parameters are tunable during optimization.
    """
    
    # Bilateral filter preprocessing (smooths while preserving edges)
    bilateral_d: int = 9  # Neighborhood diameter [5..15]
    bilateral_sigma_color: float = 75.0  # Color smoothing [50..150]
    bilateral_sigma_space: float = 75.0  # Spatial smoothing [50..150]
    
    # Canny edge detection
    sigma_px: float = 1.2  # Gaussian blur before Canny [0.8..2.4]
    hysteresis_q_low: float = 0.73  # Low threshold quantile [0.68..0.82]
    hysteresis_q_high: float = 0.92  # High threshold quantile [0.88..0.96]
    
    # Morphological operations
    closing_kernel: int = 5  # Connect nearby edges [3..9]
    merge_kernel: int = 3  # Merge fragmented edges [1..5]
    
    # Contour filtering
    min_length_px: int = 40  # Minimum contour length [10..90]
    
    # Contour linking
    link_gap_px: int = 3  # Max gap to link [0..7]
    link_angle_deg: float = 20.0  # Max angle for linking [10..30]
    
    # Vectorization
    min_len_mm: float = 3.0  # Minimum path length in mm [1.0..5.0]
    rdp_tol_mm: float = 0.30  # RDP simplification tolerance [0.15..0.60]
    
    # Dedupe (prevents double lines)
    dedupe_min_sep_px: float = 2.0  # Min separation for parallel lines [1.0..4.0]
    dedupe_max_angle_deg: float = 10.0  # Max angle for parallel detection [7..20]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EdgeParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds for each parameter.
        
        Returns
        -------
        bounds : Dict[str, Tuple[float, float]]
            Maps parameter name to (min, max) bounds
        """
        return {
            'bilateral_d': (5, 15),
            'bilateral_sigma_color': (50.0, 150.0),
            'bilateral_sigma_space': (50.0, 150.0),
            'sigma_px': (0.8, 2.4),
            'hysteresis_q_low': (0.68, 0.82),
            'hysteresis_q_high': (0.88, 0.96),
            'closing_kernel': (3, 9),
            'merge_kernel': (1, 5),
            'min_length_px': (10, 90),
            'link_gap_px': (0, 7),
            'link_angle_deg': (10.0, 30.0),
            'min_len_mm': (1.0, 5.0),
            'rdp_tol_mm': (0.15, 0.60),
            'dedupe_min_sep_px': (1.0, 4.0),
            'dedupe_max_angle_deg': (7.0, 20.0),
        }
    
    @staticmethod
    def get_int_params() -> List[str]:
        """Get list of integer-valued parameters."""
        return ['bilateral_d', 'closing_kernel', 'merge_kernel', 'min_length_px', 'link_gap_px']


@dataclass
class HatchParams:
    """Shadow hatching parameters.
    
    Controls how dark/out-of-gamut regions are filled with hatching.
    """
    
    spacing_scale: float = 2.5  # Line spacing multiplier [1.8..3.2]
    min_line_spacing_mm: float = 0.5  # Minimum line spacing [0.3..0.8]
    close_gaps_px: int = 4  # Gap closing for mask [0..8]
    min_segment_length_mm: float = 3.0  # Min segment length [1.0..4.0]
    max_hatch_coverage: float = 0.20  # Max coverage fraction [0.10..0.35]
    hatch_angle_deg: int = 45  # Hatch angle {0, 30, 45, 60, 90}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HatchParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds for each parameter."""
        return {
            'spacing_scale': (1.8, 3.2),
            'min_line_spacing_mm': (0.3, 0.8),
            'close_gaps_px': (0, 8),
            'min_segment_length_mm': (1.0, 4.0),
            'max_hatch_coverage': (0.10, 0.35),
            'hatch_angle_deg': (0, 90),  # Will be discretized to {0, 30, 45, 60, 90}
        }
    
    @staticmethod
    def get_int_params() -> List[str]:
        """Get list of integer-valued parameters."""
        return ['close_gaps_px', 'hatch_angle_deg']
    
    @staticmethod
    def get_categorical_params() -> Dict[str, List[int]]:
        """Get categorical parameter choices."""
        return {
            'hatch_angle_deg': [0, 30, 45, 60, 90]
        }


@dataclass
class GlobalParams:
    """Fixed global parameters (pen tool specs, gamut, resolution).
    
    These are NOT optimized, but passed through for consistent computation.
    """
    
    # Pen tool specifications
    tip_diameter_mm: float = 0.5  # Pen tip diameter in mm
    
    # Target resolution
    target_height_px: int = 3508  # A4 at 300 DPI
    min_px: int = 512
    max_px: int = 4096
    
    # Canvas size (will be computed from image)
    canvas_width_mm: float = 210.0  # A4 width
    canvas_height_mm: float = 297.0  # A4 height
    
    # CMY gamut bounds (simple proxy)
    cmy_min_luminance: float = 18.0  # Darkest L* achievable
    cmy_max_chroma: float = 80.0  # Max chroma
    cmy_hue_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 360.0)]  # All hues (for now)
    )
    cmy_margin: float = 0.05  # Safety margin for gamut expansion
    
    # Mode controls
    enable_edges: bool = True
    enable_hatching: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GlobalParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def compute_resolution(self, img_height: int, img_width: int) -> Tuple[int, int]:
        """Compute target resolution maintaining aspect ratio.
        
        Parameters
        ----------
        img_height : int
            Original image height
        img_width : int
            Original image width
        
        Returns
        -------
        target_height : int
            Target height (clamped to [min_px, max_px])
        target_width : int
            Target width (maintains aspect ratio)
        """
        aspect = img_width / img_height
        target_h = np.clip(self.target_height_px, self.min_px, self.max_px)
        target_w = int(target_h * aspect)
        target_w = np.clip(target_w, self.min_px, self.max_px)
        return int(target_h), int(target_w)


def save_params_yaml(path: str, edge: EdgeParams, hatch: HatchParams, glob: GlobalParams) -> None:
    """Save all parameters to YAML file.
    
    Parameters
    ----------
    path : str
        Output YAML path
    edge : EdgeParams
        Edge detection parameters
    hatch : HatchParams
        Hatching parameters
    glob : GlobalParams
        Global parameters
    """
    data = {
        'edge_params': edge.to_dict(),
        'hatch_params': hatch.to_dict(),
        'global_params': glob.to_dict(),
    }
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_params_yaml(path: str) -> Tuple[EdgeParams, HatchParams, GlobalParams]:
    """Load parameters from YAML file.
    
    Parameters
    ----------
    path : str
        Input YAML path
    
    Returns
    -------
    edge : EdgeParams
    hatch : HatchParams
    glob : GlobalParams
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    edge = EdgeParams.from_dict(data.get('edge_params', {}))
    hatch = HatchParams.from_dict(data.get('hatch_params', {}))
    glob = GlobalParams.from_dict(data.get('global_params', {}))
    
    return edge, hatch, glob


# Add numpy import for compute_resolution
import numpy as np

