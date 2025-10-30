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

