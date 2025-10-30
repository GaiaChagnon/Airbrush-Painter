"""Test YAML schema validation and config loading.

Tests for src.utils.validators:
    - Load valid example YAMLs (stroke, luts, machine, job)
    - Reject invalid YAMLs with clear error messages
    - Check error messages include line numbers and offending keys
    - Validate bounds checking (out-of-range values rejected)
    - Test schema versioning (v1, future v2)

Test cases:
    - test_load_valid_stroke_schema()
    - test_load_valid_job_config()
    - test_load_valid_machine_profile()
    - test_invalid_stroke_out_of_bounds()
    - test_invalid_stroke_missing_field()
    - test_invalid_job_nonexistent_path()
    - test_lut_metadata_validation()

Fixtures:
    - fixtures/valid_stroke.yaml
    - fixtures/invalid_stroke_*.yaml
    - fixtures/valid_job.yaml
    - fixtures/invalid_job_*.yaml

Run:
    pytest tests/test_schemas.py -v
"""

