"""SHA-256 hashing for file and tensor provenance.

Provides:
    - sha256_file(): Hash file contents (for LUTs, datasets, images)
    - sha256_tensor(): Hash tensor values (for checkpoints, calibration data)

Used for MLflow provenance:
    - Log LUT hashes to detect calibration changes
    - Log validation set hash to ensure consistency across HPO trials
    - Cache keys for expensive computations

Deterministic hashing:
    - Tensors converted to bytes via .cpu().numpy().tobytes()
    - Files read in chunks (1 MB default) for memory efficiency
    - Results are hex strings (64 chars)

Example MLflow params:
    color_lut_sha256: "a3f5b2..."
    validation_set_sha256: "7d1e4c..."

Usage:
    from src.utils import hashing
    lut_hash = hashing.sha256_file("configs/sim/luts/color_lut.pt")

Note: Module renamed from `hash.py` to `hashing.py` to avoid shadowing builtin `hash()`.
"""

import hashlib
from pathlib import Path
from typing import Union

import torch


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of file contents.

    Parameters
    ----------
    path : Union[str, Path]
        File path
    chunk_size : int
        Read chunk size in bytes, default 1 MB

    Returns
    -------
    str
        SHA-256 hex digest (64 characters)

    Raises
    ------
    FileNotFoundError
        If file doesn't exist

    Examples
    --------
    >>> lut_hash = sha256_file("configs/sim/luts/color_lut.pt")
    >>> print(f"color_lut_sha256: {lut_hash}")
    color_lut_sha256: a3f5b2c1d4e7f8...
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256 = hashlib.sha256()
    
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    
    return sha256.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    """Compute SHA-256 hash of tensor values.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to hash (any shape, dtype)

    Returns
    -------
    str
        SHA-256 hex digest (64 characters)

    Notes
    -----
    Deterministic: same tensor values → same hash.
    Moves tensor to CPU and converts to bytes for hashing.
    Hash is invariant to device but NOT to dtype/shape.

    Examples
    --------
    >>> lut = torch.load("configs/sim/luts/color_lut.pt")
    >>> lut_hash = sha256_tensor(lut)
    >>> mlflow.log_param("color_lut_hash", lut_hash)
    """
    # Move to CPU and convert to numpy bytes
    t_bytes = t.detach().cpu().numpy().tobytes()
    
    sha256 = hashlib.sha256()
    sha256.update(t_bytes)
    
    return sha256.hexdigest()


def sha256_string(s: str) -> str:
    """Compute SHA-256 hash of string.

    Parameters
    ----------
    s : str
        String to hash

    Returns
    -------
    str
        SHA-256 hex digest (64 characters)

    Examples
    --------
    >>> config_hash = sha256_string(str(config_dict))
    """
    sha256 = hashlib.sha256()
    sha256.update(s.encode('utf-8'))
    return sha256.hexdigest()


def hash_directory_contents(
    root: Union[str, Path],
    pattern: str = "*"
) -> str:
    """Compute aggregate hash of directory contents.

    Parameters
    ----------
    root : Union[str, Path]
        Root directory
    pattern : str
        Glob pattern for files to include, default "*"

    Returns
    -------
    str
        SHA-256 hex digest of concatenated file hashes (sorted by path)

    Notes
    -----
    Used for validation set hashing (entire directory).
    Files processed in sorted order for determinism.

    Examples
    --------
    >>> val_hash = hash_directory_contents("data/validation_images/cmy_only/", "*.png")
    >>> mlflow.log_param("validation_set_sha256", val_hash)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    
    # Collect all matching files in sorted order
    files = sorted(root.rglob(pattern))
    
    # Concatenate individual file hashes
    combined = hashlib.sha256()
    for file_path in files:
        if file_path.is_file():
            file_hash = sha256_file(file_path)
            combined.update(file_hash.encode('utf-8'))
    
    return combined.hexdigest()


def hash_dict(d: dict) -> str:
    """Compute SHA-256 hash of dictionary (sorted keys).

    Parameters
    ----------
    d : dict
        Dictionary to hash (must be JSON-serializable)

    Returns
    -------
    str
        SHA-256 hex digest

    Notes
    -----
    Sorts keys for determinism.
    Used for config hashing.

    Examples
    --------
    >>> config_hash = hash_dict(config)
    >>> mlflow.log_param("config_sha256", config_hash)
    """
    import json
    
    # Sort keys for determinism
    json_str = json.dumps(d, sort_keys=True)
    return sha256_string(json_str)


def verify_file_hash(path: Union[str, Path], expected_hash: str) -> bool:
    """Verify file matches expected hash.

    Parameters
    ----------
    path : Union[str, Path]
        File path
    expected_hash : str
        Expected SHA-256 hex digest

    Returns
    -------
    bool
        True if hash matches, False otherwise

    Examples
    --------
    >>> if not verify_file_hash("color_lut.pt", expected_hash):
    ...     print("WARNING: LUT file modified!")
    """
    actual_hash = sha256_file(path)
    return actual_hash == expected_hash


def compute_lut_provenance(lut_dir: Union[str, Path]) -> dict:
    """Compute provenance hashes for all LUT files.

    Parameters
    ----------
    lut_dir : Union[str, Path]
        Directory containing LUT files (*.pt)

    Returns
    -------
    dict
        Dictionary mapping LUT name to SHA-256 hash

    Examples
    --------
    >>> lut_provenance = compute_lut_provenance("configs/sim/luts/")
    >>> mlflow.log_params(lut_provenance)
    {'color_lut_sha256': 'a3f5b2...', 'psf_lut_sha256': '7d1e4c...', ...}
    """
    lut_dir = Path(lut_dir)
    provenance = {}
    
    for lut_file in lut_dir.glob("*.pt"):
        lut_name = lut_file.stem
        lut_hash = sha256_file(lut_file)
        provenance[f"{lut_name}_sha256"] = lut_hash
    
    return provenance
