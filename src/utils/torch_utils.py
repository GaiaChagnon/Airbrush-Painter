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

