"""Cross-cutting utilities (lowest dependency layer).

This package provides shared primitives for:
    - Config validation (validators)
    - Coordinate conversions & LPIPS tiling (compute)
    - Color science (color)
    - Geometry operations (geometry)
    - Atomic I/O (fs)
    - Torch ergonomics (torch_utils)
    - Metrics (metrics)
    - Profiling (profiler)
    - Stroke helpers (strokes)
    - Hashing for provenance (hashing)
    - Unified logging (logging_config)
    - MLflow integration (mlflow_helpers)

No module in utils/ may import from upper layers (env, rl_agent, simulator, etc.).

Convenience imports:
    from src.utils import fs, compute, color
    from src.utils.logging_config import setup_logging, get_logger
"""

# Re-export commonly used modules for convenience
from . import color
from . import compute
from . import fs
from . import geometry
from . import hashing
from . import logging_config
from . import metrics
from . import profiler
from . import strokes
from . import torch_utils
from . import validators

# Common functions for direct import
from .logging_config import get_logger, push_context, setup_logging

__all__ = [
    # Modules
    'color',
    'compute',
    'fs',
    'geometry',
    'hashing',
    'logging_config',
    'metrics',
    'profiler',
    'strokes',
    'torch_utils',
    'validators',
    # Direct exports
    'setup_logging',
    'get_logger',
    'push_context',
]
