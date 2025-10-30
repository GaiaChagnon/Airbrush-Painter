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
    - G-code generation (gcode_generator)
    - G-code simulation (gcode_vm)

No module in utils/ may import from upper layers (env, rl_agent, simulator, etc.).

Convenience imports:
    from src.utils import fs, compute, color, validators
    from src.utils.logging_config import setup_logging, get_logger
"""

# Re-export commonly used modules for convenience
from . import color
from . import compute
from . import fs
from . import gcode_generator
from . import gcode_vm
from . import geometry
from . import hashing
from . import logging_config
from . import metrics
from . import mlflow_helpers
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
    'gcode_generator',
    'gcode_vm',
    'geometry',
    'hashing',
    'logging_config',
    'metrics',
    'mlflow_helpers',
    'profiler',
    'strokes',
    'torch_utils',
    'validators',
    # Direct exports
    'setup_logging',
    'get_logger',
    'push_context',
]
