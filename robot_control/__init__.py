"""
Robot Control Package.

Isolated control system for Klipper-based pen/airbrush robot. Communicates with
Klipper via Unix Domain Socket API for motion control, calibration, and job execution.

Subpackages:
    hardware: Klipper client, job executor, interactive controls
    job_ir: Intermediate representation for job operations
    calibration: Test patterns and calibration routines
    gcode: G-code generation from Job IR
    configs: Machine configuration loading and validation
"""

__all__ = ["hardware", "job_ir", "calibration", "gcode", "configs"]
