"""Lightweight profiling: wall-clock timers and NVTX markers.

Provides:
    - timer(): Context manager for wall-clock timing with optional sink
    - nvtx_range(): NVIDIA Nsight markers for GPU profiling

Used to measure:
    - Strategist forward pass
    - Technician refinement steps
    - Renderer (rasterization + compositing)
    - LPIPS computation
    - G-code generation

Timings logged to MLflow for performance tracking across trials.

NVTX ranges visible in:
    - NVIDIA Nsight Systems (timeline)
    - NVIDIA Nsight Compute (kernel analysis)

No heavy dependencies (no line_profiler, cProfile overhead during training).
"""

