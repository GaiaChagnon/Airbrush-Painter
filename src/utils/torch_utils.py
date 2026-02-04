"""PyTorch ergonomics: seeding, device management, precision, memory.

Provides:
    - seed_everything(): Reproducible training (torch, numpy, Python RNG)
    - Device helpers: to_device_recursive() for nested structures
    - Channels-last: set_channels_last() for CNN performance (free on DGX)
    - Precision contexts: autocast_context() with BF16/FP16/FP32
    - Memory profiling: get_gpu_mem_highwater() for MLflow metrics

DGX Spark defaults:
    - BF16 (Blackwell Tensor Cores) for networks
    - FP32 for LUTs and LPIPS
    - Channels-last memory format for CNNs
    - pin_memory=False (UMA architecture)
    - grad_checkpointing=False (ample memory, prioritize speed)

Reproducibility:
    Logs PYTHONHASHSEED, torch seeds, and cudnn flags to MLflow
    for exact run replication.
"""

import os
import random
from typing import Any, Dict, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn


def cuda_is_functional() -> bool:
    """Check if CUDA is available and can actually execute operations.
    
    Returns
    -------
    bool
        True if CUDA is available and functional, False otherwise.
        
    Notes
    -----
    This function tests actual CUDA execution, not just availability.
    Useful for skipping GPU tests when hardware is incompatible (e.g.,
    compute capability mismatch).
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try a simple operation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            t = torch.tensor([1.0], device='cuda')
            _ = t + 1.0
            torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> Dict[str, Any]:
    """Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    deterministic_cudnn : bool
        If True, set cudnn.deterministic=True and benchmark=False
        Slower but reproducible. Default True.

    Returns
    -------
    Dict[str, Any]
        Dictionary of seed settings for logging to MLflow:
        - seed: int
        - PYTHONHASHSEED: str
        - torch_version: str
        - cuda_available: bool
        - cudnn_deterministic: bool
        - cudnn_benchmark: bool

    Notes
    -----
    Sets seeds for:
        - Python random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA (all devices)
        - PYTHONHASHSEED environment variable

    Warning: deterministic_cudnn=True can significantly slow training.
    For production HPO, consider False with multiple seeds.

    Examples
    --------
    >>> seed_info = seed_everything(42)
    >>> mlflow.log_params(seed_info)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set PYTHONHASHSEED for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure cudnn for determinism
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic_cudnn
        torch.backends.cudnn.benchmark = not deterministic_cudnn
    
    # Return settings for logging
    return {
        'seed': seed,
        'PYTHONHASHSEED': str(seed),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': deterministic_cudnn,
        'cudnn_benchmark': not deterministic_cudnn
    }


def to_device_recursive(
    obj: Any,
    device: torch.device,
    non_blocking: bool = True
) -> Any:
    """Move nested structure to device recursively.

    Parameters
    ----------
    obj : Any
        Object to move (tensor, dict, list, tuple, or nested)
    device : torch.device
        Target device
    non_blocking : bool
        Use non-blocking transfer (requires pinned memory), default True

    Returns
    -------
    Any
        Object moved to device (same structure)

    Notes
    -----
    Handles:
        - torch.Tensor → .to(device)
        - nn.Module → .to(device)
        - dict → recurse on values
        - list/tuple → recurse on elements
        - primitives → return as-is

    Examples
    --------
    >>> batch = {'obs': torch.randn(1, 9, 256, 256), 'mask': torch.ones(1, 1, 256, 256)}
    >>> batch_gpu = to_device_recursive(batch, torch.device('cuda'))
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device_recursive(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device_recursive(item, device, non_blocking) for item in obj)
    else:
        return obj


def set_channels_last(model: nn.Module) -> None:
    """Convert model to channels-last memory format (in-place).

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    Notes
    -----
    Channels-last is often faster for CNNs on modern GPUs (DGX Spark).
    Converts all 4D parameters/buffers (weights) to channels-last.
    Free performance win for convolution-heavy models.

    Must also convert inputs to channels-last in forward:
        x = x.to(memory_format=torch.channels_last)

    Examples
    --------
    >>> model = ResNet34()
    >>> set_channels_last(model)
    >>> x = torch.randn(1, 3, 256, 256).to(memory_format=torch.channels_last)
    >>> out = model(x)
    """
    model = model.to(memory_format=torch.channels_last)
    
    # Also convert all existing parameters/buffers
    for name, param in model.named_parameters():
        if param.ndim == 4:  # Conv weights
            param.data = param.data.to(memory_format=torch.channels_last)
    
    for name, buf in model.named_buffers():
        if buf.ndim == 4:
            buf.data = buf.data.to(memory_format=torch.channels_last)


def get_gpu_mem_highwater(reset: bool = True) -> int:
    """Get peak GPU memory usage since last reset.

    Parameters
    ----------
    reset : bool
        If True, reset the peak tracker after reading, default True

    Returns
    -------
    int
        Peak memory in bytes

    Notes
    -----
    Returns 0 if CUDA not available.
    Useful for logging to MLflow as performance metric.

    Examples
    --------
    >>> peak_mem_mb = get_gpu_mem_highwater() / (1024**2)
    >>> mlflow.log_metric("peak_gpu_mem_mb", peak_mem_mb)
    """
    if not torch.cuda.is_available():
        return 0
    
    peak_mem = torch.cuda.max_memory_allocated()
    
    if reset:
        torch.cuda.reset_peak_memory_stats()
    
    return peak_mem


def get_gpu_mem_info() -> Dict[str, int]:
    """Get current GPU memory statistics.

    Returns
    -------
    Dict[str, int]
        Dictionary with keys:
        - allocated: Currently allocated memory (bytes)
        - reserved: Memory reserved by PyTorch (bytes)
        - free: Free memory (bytes)
        - total: Total GPU memory (bytes)
        Returns zeros if CUDA not available

    Examples
    --------
    >>> mem_info = get_gpu_mem_info()
    >>> print(f"Allocated: {mem_info['allocated'] / 1e9:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0,
            'reserved': 0,
            'free': 0,
            'total': 0
        }
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def no_autocast():
    """Context manager to disable autocast (for LPIPS, etc.).

    Yields
    ------
    Context with autocast disabled

    Notes
    -----
    Use for operations requiring FP32 precision (LPIPS, loss computation).

    Examples
    --------
    >>> with no_autocast():
    ...     loss = lpips_fn(img1.float(), img2.float())
    """
    # Use torch.cuda.amp.autocast with enabled=False
    return torch.cuda.amp.autocast(enabled=False)


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """Calculate model size and parameter counts.

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    Returns
    -------
    Dict[str, int]
        Dictionary with:
        - total_params: Total parameter count
        - trainable_params: Trainable parameter count
        - total_size_mb: Total size in MB (FP32 equivalent)

    Examples
    --------
    >>> model_info = get_model_size(model)
    >>> print(f"Trainable: {model_info['trainable_params']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Size in bytes (assuming FP32)
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size_bytes / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_size_mb': total_size_mb
    }


def print_model_summary(model: nn.Module, name: str = "Model") -> None:
    """Print concise model summary.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    name : str
        Model name for display

    Notes
    -----
    Prints parameter counts and memory footprint.
    """
    info = get_model_size(model)
    
    print(f"\n{name} Summary:")
    print(f"  Total params:     {info['total_params']:,}")
    print(f"  Trainable params: {info['trainable_params']:,}")
    print(f"  Model size:       {info['total_size_mb']:.2f} MB")
    
    # Check channels-last
    has_channels_last = False
    for param in model.parameters():
        if param.ndim == 4 and param.is_contiguous(memory_format=torch.channels_last):
            has_channels_last = True
            break
    
    if has_channels_last:
        print(f"  Memory format:    channels_last")
    else:
        print(f"  Memory format:    contiguous")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    trainable_only : bool
        If True, count only trainable parameters, default False

    Returns
    -------
    int
        Parameter count

    Examples
    --------
    >>> num_params = count_parameters(model, trainable_only=True)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def enable_tf32() -> None:
    """Enable TF32 for faster training on Ampere+ GPUs.

    Notes
    -----
    TF32 provides ~10x speedup for matmul/conv with minimal precision loss.
    Enabled by default on PyTorch 1.12+, but can be explicitly set.
    Safe for most training (not for LPIPS or critical numerics).
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def disable_tf32() -> None:
    """Disable TF32 for strict FP32 precision.

    Notes
    -----
    Use when numerical precision is critical.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def empty_cache() -> None:
    """Release cached GPU memory.

    Notes
    -----
    Useful after OOM to free unused cached memory.
    Does not free allocated memory (only cache).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    """Synchronize CUDA operations (wait for completion).

    Notes
    -----
    Used for accurate timing measurements.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
