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

import time
from contextlib import contextmanager
from typing import Callable, Optional


@contextmanager
def timer(name: str, sink: Optional[Callable[[str, float], None]] = None):
    """Context manager for wall-clock timing.

    Parameters
    ----------
    name : str
        Timer name (for logging/display)
    sink : Optional[Callable[[str, float], None]]
        Optional callback(name, elapsed_seconds)
        If None, prints to stdout

    Yields
    ------
    None

    Examples
    --------
    >>> with timer("forward_pass"):
    ...     output = model(input)
    forward_pass: 0.123 s

    >>> def log_to_mlflow(name, elapsed):
    ...     mlflow.log_metric(f"time_{name}", elapsed)
    >>> with timer("lpips", sink=log_to_mlflow):
    ...     loss = lpips(img1, img2)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if sink is not None:
            sink(name, elapsed)
        else:
            print(f"{name}: {elapsed:.3f} s")


@contextmanager
def nvtx_range(msg: str, color: int = 0x00AAFF):
    """Context manager for NVIDIA NVTX range markers.

    Parameters
    ----------
    msg : str
        Range label (visible in Nsight Systems timeline)
    color : int
        RGB color code (hex), default 0x00AAFF (blue)

    Yields
    ------
    None

    Notes
    -----
    No-op if CUDA not available or nvtx not installed.
    Visible in Nsight Systems/Compute profiling tools.

    Examples
    --------
    >>> with nvtx_range("Technician refinement", color=0xFF0000):
    ...     action = technician(canvas, target)
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(msg)
    except (ImportError, AttributeError):
        pass
    
    try:
        yield
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except (ImportError, AttributeError):
            pass


class TimerAccumulator:
    """Accumulate multiple timing measurements for averaging.

    Attributes
    ----------
    name : str
        Timer name
    total_time : float
        Accumulated time in seconds
    count : int
        Number of measurements

    Methods
    -------
    measure() : Context manager for single measurement
    mean() : Get mean time per measurement
    reset() : Clear accumulated data

    Examples
    --------
    >>> strategist_timer = TimerAccumulator("strategist")
    >>> for _ in range(100):
    ...     with strategist_timer.measure():
    ...         action = strategist(obs)
    >>> print(f"Mean: {strategist_timer.mean():.4f} s")
    """

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.count = 0

    @contextmanager
    def measure(self):
        """Context manager to measure and accumulate time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.total_time += elapsed
            self.count += 1

    def mean(self) -> float:
        """Get mean time per measurement.

        Returns
        -------
        float
            Mean time in seconds, or 0.0 if no measurements
        """
        return self.total_time / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        """Reset accumulated data."""
        self.total_time = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return f"TimerAccumulator({self.name}, mean={self.mean():.4f}s, count={self.count})"


def synchronize_and_time(fn: Callable, *args, **kwargs) -> tuple:
    """Execute function with CUDA synchronization for accurate timing.

    Parameters
    ----------
    fn : Callable
        Function to time
    *args, **kwargs
        Arguments to fn

    Returns
    -------
    tuple
        (result, elapsed_seconds)

    Notes
    -----
    Synchronizes CUDA before and after execution for accurate GPU timing.
    Use for micro-benchmarks; adds overhead.

    Examples
    --------
    >>> result, elapsed = synchronize_and_time(model, input_tensor)
    >>> print(f"Inference: {elapsed*1000:.2f} ms")
    """
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return result, elapsed
