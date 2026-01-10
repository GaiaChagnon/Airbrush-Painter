"""
G-code generation module.

Converts Job IR operations to G-code strings with coordinate transforms
and Z state mapping.
"""

from robot_control.gcode.generator import GCodeGenerator

__all__ = ["GCodeGenerator"]
