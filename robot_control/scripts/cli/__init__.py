"""Unified robot CLI package.

Provides a Rich-rendered terminal interface for all robot control
modes: interactive jog, pump controller, lineart tracer, and
calibration.  Each mode is isolated in its own module and exposes
a ``run(app) -> None`` entry point.
"""

from __future__ import annotations

__all__ = ["RobotApp"]

# Lazy import to avoid circular dependency; resolved at first attribute access.
def __getattr__(name: str) -> object:
    if name == "RobotApp":
        from robot_control.scripts.cli.app import RobotApp as _RobotApp
        return _RobotApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
