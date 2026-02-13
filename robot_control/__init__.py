"""Robot control package for Klipper-based pen/airbrush machine.

Subpackages:
    configs     -- Machine YAML config loading and validation
    hardware    -- Klipper API client, job executor, interactive controller
    job_ir      -- Job IR operation dataclasses
    gcode       -- Job IR to G-code conversion
    calibration -- Test patterns and guided calibration routines
    scripts     -- Thin CLI entry points
"""

__all__ = [
    "calibration",
    "configs",
    "gcode",
    "hardware",
    "job_ir",
]
