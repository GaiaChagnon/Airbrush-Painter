"""Atomic filesystem operations for safe file writes and YAML handling.

Provides:
    - Atomic writes: tmp file → fsync → rename (prevents partial reads)
    - YAML load/save with validation
    - Atomic symlink creation for "latest" pointers
    - Directory creation with exist_ok semantics
    - Epoch discovery for training monitor

Critical for decoupled GUI:
    - Training writes artifacts atomically
    - GUI reads via watchdog without seeing partial files
    - Symlinks point to latest complete epoch

All paths use pathlib.Path for cross-platform compatibility.
YAML files use ruamel.yaml for round-trip preservation of comments.

Usage:
    from src.utils import fs
    fs.atomic_save_image(canvas_np, epoch_dir / "canvas.png")
    fs.atomic_yaml_dump(metadata, epoch_dir / "metadata.yaml")
    fs.symlink_atomic(epoch_dir, "outputs/training_monitor/latest")

Note: Module renamed from `io.py` to `fs.py` to avoid shadowing stdlib `io`.
"""

