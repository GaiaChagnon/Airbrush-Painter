"""Test hashing functions for provenance.

Tests for src.utils.hash:
    - sha256_file() produces consistent hashes
    - sha256_tensor() produces consistent hashes
    - Different files → different hashes
    - Same tensor on CPU/GPU → same hash

Test cases:
    - test_sha256_file_consistent()
    - test_sha256_file_different()
    - test_sha256_tensor_consistent()
    - test_sha256_tensor_cpu_gpu_same()
    - test_sha256_tensor_different()

Known hash test:
    - Create file with known content
    - Verify hash matches expected (hex string, 64 chars)

Run:
    pytest tests/test_hash.py -v
"""

