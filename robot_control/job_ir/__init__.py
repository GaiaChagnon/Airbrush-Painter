"""
Job Intermediate Representation module.

Defines all possible job operations as immutable dataclasses. This vocabulary
is the contract between vector data and G-code generation.

All coordinates are in millimeters, canvas-relative.
"""

from robot_control.job_ir.operations import (
    Operation,
    HomeXY,
    SelectTool,
    RapidXY,
    LinearMove,
    DrawPolyline,
    ToolUp,
    ToolDown,
    Stroke,
    Job,
)

__all__ = [
    "Operation",
    "HomeXY",
    "SelectTool",
    "RapidXY",
    "LinearMove",
    "DrawPolyline",
    "ToolUp",
    "ToolDown",
    "Stroke",
    "Job",
]
