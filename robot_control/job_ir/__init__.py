"""Job IR -- intermediate representation for painting jobs.

All operations are frozen dataclasses with canvas-relative mm coordinates.
"""

from robot_control.job_ir.operations import (
    DrawPolyline,
    HomeXY,
    Job,
    LinearMove,
    Operation,
    Purge,
    RapidXY,
    RefillPump,
    SelectTool,
    SetInkMix,
    SprayOff,
    SprayOn,
    Stroke,
    ToolDown,
    ToolUp,
    create_stroke,
    operations_to_strokes,
)

__all__ = [
    "DrawPolyline",
    "HomeXY",
    "Job",
    "LinearMove",
    "Operation",
    "Purge",
    "RapidXY",
    "RefillPump",
    "SelectTool",
    "SetInkMix",
    "SprayOff",
    "SprayOn",
    "Stroke",
    "ToolDown",
    "ToolUp",
    "create_stroke",
    "operations_to_strokes",
]
