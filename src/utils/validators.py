"""YAML schema validation and config loading.

Provides centralized validation for all configuration files:
    - Stroke schema (stroke.v1.yaml): Bézier parameters, z/speed profiles, CMY color
    - LUT schema (luts.v1.yaml): Color/PSF/alpha lookup table metadata
    - Machine schema (machine.v1.yaml): Work area, feeds, G-code flavor, macros
    - Job schema (job.v1.yaml): Complete job specification for inference

All modules must use these validators to load configs for fail-fast error detection
with actionable messages (line numbers, offending keys, expected ranges).

Public API:
    load_job_config(path) -> JobV1
    load_machine_profile(path) -> MachineV1
    load_stroke_schema(path) -> StrokeSchemaV1
    validate_strokes(path) -> None
    validate_luts(path) -> None

Units:
    - Geometry: millimeters (mm)
    - Speed: mm/s
    - Color: [0.0, 1.0] for CMY and RGB
"""

