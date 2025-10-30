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
"""

