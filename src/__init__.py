"""Airbrush Painter: robotic painting system.

Core modules for simulating airbrush paint physics, calibrating the
renderer against real hardware, and generating G-code for a
Klipper-controlled gantry.

Architecture layers (strict one-way dependency):
    scripts/ → src/{airbrush_simulator,data_pipeline}/ → src/utils/

Key invariants:
    - Geometry in millimeters end-to-end
    - YAML-only configs, no JSON
    - All images are linear RGB [0,1] unless explicitly noted
"""

__version__ = "3.0.0"

