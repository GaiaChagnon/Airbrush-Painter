"""Test PyTorch utility functions.

Tests for src.utils.torch_utils:
    - seed_everything() produces reproducible results
    - to_device_recursive() handles nested structures
    - set_channels_last() no-op (doesn't break model)
    - get_gpu_mem_highwater() returns reasonable value
    - autocast_context() works for BF16/FP16/FP32

Test cases:
    - test_seed_everything_reproducible()
    - test_to_device_recursive()
    - test_set_channels_last()
    - test_get_gpu_mem_highwater()
    - test_autocast_context_bf16()
    - test_autocast_context_fp32()

Reproducibility test:
    - Seed with value A
    - Generate random tensors
    - Seed again with value A
    - Generate again
    - Assert: tensors are identical

Run:
    pytest tests/test_torch_utils.py -v
"""

