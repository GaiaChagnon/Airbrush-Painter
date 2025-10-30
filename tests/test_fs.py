"""Test atomic filesystem operations.

Tests for src.utils.fs:
    - Atomic writes prevent partial reads (concurrent access)
    - YAML roundtrip preserves structure
    - Symlink_atomic replaces existing symlinks
    - Epoch discovery from directory names
    - ensure_dir creates parents

Test cases:
    - test_atomic_write_bytes()
    - test_atomic_save_image()
    - test_atomic_yaml_dump()
    - test_load_yaml_roundtrip()
    - test_symlink_atomic()
    - test_symlink_atomic_replaces_existing()
    - test_find_latest_epoch()
    - test_ensure_dir()

Concurrent access test:
    - Writer: atomic_write_bytes()
    - Reader: Read in tight loop
    - Assert: Reader never sees partial data (read succeeds or ENOENT, never corrupt)

Run:
    pytest tests/test_fs.py -v
"""

