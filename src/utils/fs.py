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

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image


def ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, return Path object.

    Parameters
    ----------
    p : Union[str, Path]
        Directory path

    Returns
    -------
    Path
        Path object (guaranteed to exist)

    Notes
    -----
    Thread-safe (mkdir with exist_ok=True).
    Creates parent directories as needed.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(
    path: Union[str, Path],
    data: bytes,
    tmp_suffix: str = ".tmp"
) -> None:
    """Write bytes to file atomically (tmp → fsync → rename).

    Parameters
    ----------
    path : Union[str, Path]
        Target file path
    data : bytes
        Data to write
    tmp_suffix : str
        Temporary file suffix, default ".tmp"

    Notes
    -----
    Prevents GUI from reading partial files during training writes.
    Uses same directory for tmp file to ensure atomic rename on same filesystem.
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    # Create temporary file in same directory
    tmp_path = path.with_suffix(path.suffix + tmp_suffix)
    
    try:
        # Write to temporary file
        with open(tmp_path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Ensure data reaches disk
        
        # Atomic rename (overwrites existing file on POSIX)
        tmp_path.replace(path)
    except Exception as e:
        # Clean up tmp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to write {path} atomically: {e}") from e


def atomic_save_image(
    img: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    pil_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """Save image atomically (prevents partial reads).

    Parameters
    ----------
    img : Union[np.ndarray, torch.Tensor]
        Image data:
        - numpy: (H, W, 3) uint8 or (H, W) uint8
        - torch: (3, H, W) or (H, W) float [0,1]
    path : Union[str, Path]
        Target file path (extension determines format)
    pil_kwargs : Optional[Dict[str, Any]]
        Additional kwargs for PIL.Image.save (e.g., quality=95)

    Notes
    -----
    Converts torch tensors to numpy uint8 before saving.
    Supports PNG, JPEG, etc. (via PIL).
    """
    path = Path(path)
    pil_kwargs = pil_kwargs or {}
    
    # Convert to numpy uint8
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            # (C, H, W) → (H, W, C)
            img = img.permute(1, 2, 0)
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            # [0, 1] float → [0, 255] uint8
            img = (img.clamp(0, 1) * 255).byte()
        img = img.numpy()
    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Squeeze single-channel to (H, W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)
    
    # Save via PIL
    pil_img = Image.fromarray(img)
    
    # Use atomic write
    # Create tmp file with same extension to preserve format detection
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        pil_img.save(tmp_path, **pil_kwargs)
        tmp_path.replace(path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to save image {path} atomically: {e}") from e


def atomic_yaml_dump(obj: Any, path: Union[str, Path]) -> None:
    """Save object as YAML atomically.

    Parameters
    ----------
    obj : Any
        Python object (dict, list, primitives)
    path : Union[str, Path]
        Target YAML file path

    Notes
    -----
    Uses PyYAML safe_dump for security.
    Preserves ordering (Python 3.7+).
    """
    path = Path(path)
    
    # Serialize to string
    yaml_str = yaml.safe_dump(
        obj,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True
    )
    
    # Write atomically
    atomic_write_bytes(path, yaml_str.encode('utf-8'))


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file safely.

    Parameters
    ----------
    path : Union[str, Path]
        YAML file path

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    yaml.YAMLError
        If YAML parsing fails

    Notes
    -----
    Uses safe_load to prevent arbitrary code execution.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {path}: {e}") from e


def symlink_atomic(target: Union[str, Path], link_path: Union[str, Path]) -> None:
    """Create or update symlink atomically.

    Parameters
    ----------
    target : Union[str, Path]
        Symlink target (can be relative or absolute)
    link_path : Union[str, Path]
        Symlink path to create

    Notes
    -----
    Uses tmp symlink + rename for atomicity.
    Overwrites existing symlink safely.
    Creates parent directories as needed.
    """
    target = Path(target)
    link_path = Path(link_path)
    ensure_dir(link_path.parent)
    
    # Create temporary symlink
    tmp_link = link_path.with_suffix(".tmp_symlink")
    
    try:
        # Remove tmp link if exists
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        
        # Create symlink (relative or absolute)
        if not target.is_absolute():
            # Make relative to link_path parent
            tmp_link.symlink_to(target)
        else:
            tmp_link.symlink_to(target)
        
        # Atomic replace
        tmp_link.replace(link_path)
    except Exception as e:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        raise RuntimeError(f"Failed to create symlink {link_path} → {target}: {e}") from e


def make_latest_symlink(dir_to_link: Union[str, Path], latest_link: Union[str, Path]) -> None:
    """Create "latest" symlink pointing to directory.

    Parameters
    ----------
    dir_to_link : Union[str, Path]
        Directory to point to
    latest_link : Union[str, Path]
        Path for "latest" symlink

    Notes
    -----
    Convenience wrapper around symlink_atomic.
    Used for training monitor: outputs/training_monitor/latest → epoch_N
    """
    symlink_atomic(dir_to_link, latest_link)


def find_latest_epoch(root: Union[str, Path]) -> Optional[int]:
    """Find latest epoch directory in training monitor.

    Parameters
    ----------
    root : Union[str, Path]
        Root directory to search (e.g., outputs/training_monitor)

    Returns
    -------
    Optional[int]
        Latest epoch number, or None if no epochs found

    Notes
    -----
    Searches for directories matching "epoch_N" pattern.
    Returns maximum N found.
    Used by GUI to populate epoch selector.
    """
    root = Path(root)
    if not root.exists():
        return None
    
    epoch_nums = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("epoch_"):
            try:
                epoch_num = int(item.name.split("_")[1])
                epoch_nums.append(epoch_num)
            except (ValueError, IndexError):
                continue
    
    return max(epoch_nums) if epoch_nums else None


def list_epoch_dirs(root: Union[str, Path]) -> list:
    """List all epoch directories sorted by epoch number.

    Parameters
    ----------
    root : Union[str, Path]
        Root directory (e.g., outputs/training_monitor)

    Returns
    -------
    list[Path]
        Sorted list of epoch directories

    Notes
    -----
    Used by GUI epoch selector.
    """
    root = Path(root)
    if not root.exists():
        return []
    
    epochs = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("epoch_"):
            try:
                epoch_num = int(item.name.split("_")[1])
                epochs.append((epoch_num, item))
            except (ValueError, IndexError):
                continue
    
    # Sort by epoch number
    epochs.sort(key=lambda x: x[0])
    return [path for _, path in epochs]


def atomic_write_text(
    path: Union[str, Path],
    text: str,
    encoding: str = "utf-8"
) -> None:
    """Write text to file atomically.

    Parameters
    ----------
    path : Union[str, Path]
        Target file path
    text : str
        Text content
    encoding : str
        Text encoding, default "utf-8"

    Notes
    -----
    Convenience wrapper around atomic_write_bytes.
    """
    atomic_write_bytes(path, text.encode(encoding))


def safe_remove(path: Union[str, Path]) -> bool:
    """Remove file or symlink safely (no error if missing).

    Parameters
    ----------
    path : Union[str, Path]
        Path to remove

    Returns
    -------
    bool
        True if removed, False if didn't exist

    Notes
    -----
    Used for cleanup operations.
    """
    path = Path(path)
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
            return True
        return False
    except Exception:
        return False
