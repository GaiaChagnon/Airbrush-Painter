"""Measurement input and calculation helpers for calibration routines.

Provides interactive user-prompt functions and the math needed to turn
raw measurements into configuration corrections.

All distance values are in **millimetres**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MeasurementResult:
    """Result of a calibration measurement.

    Parameters
    ----------
    commanded : float
        Distance the machine was told to move (mm).
    measured : float
        Distance the user actually measured (mm).
    error : float
        ``commanded - measured`` (mm).
    error_percent : float
        ``error / commanded * 100``.
    correction_factor : float
        ``commanded / measured`` -- multiply the old config value by
        this to get the corrected value.
    """

    commanded: float
    measured: float
    error: float
    error_percent: float
    correction_factor: float


# ---------------------------------------------------------------------------
# Interactive input helpers
# ---------------------------------------------------------------------------


def get_float_input(prompt: str, default: float | None = None) -> float:
    """Prompt the user for a float value.

    Parameters
    ----------
    prompt : str
        Message displayed.
    default : float | None
        Value returned if the user presses Enter without typing.

    Returns
    -------
    float
        Validated user input.
    """
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid number: %r -- try again", raw)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer.

    Parameters
    ----------
    prompt : str
        Question text.
    default : bool
        Default if Enter is pressed alone.

    Returns
    -------
    bool
    """
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        logger.warning("Please enter y or n")


def get_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Prompt the user to pick from a list.

    Parameters
    ----------
    prompt : str
        Question text.
    choices : list[str]
        Valid option strings.
    default : str | None
        Default if Enter is pressed.

    Returns
    -------
    str
        One of *choices*.
    """
    hint = ", ".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt} ({hint}){suffix}: ").strip()
        if not raw and default is not None:
            return default
        if raw in choices:
            return raw
        logger.warning("Invalid choice: %r -- pick from %s", raw, hint)


# ---------------------------------------------------------------------------
# Calculation helpers
# ---------------------------------------------------------------------------


def calculate_steps_correction(
    commanded: float,
    measured: float,
) -> MeasurementResult:
    """Compute the correction factor for a steps-per-mm calibration.

    Parameters
    ----------
    commanded : float
        Distance the machine was told to move (mm).
    measured : float
        Distance the user measured (mm).

    Returns
    -------
    MeasurementResult
    """
    if measured <= 0:
        raise ValueError(f"measured must be > 0, got {measured}")
    error = commanded - measured
    return MeasurementResult(
        commanded=commanded,
        measured=measured,
        error=error,
        error_percent=(error / commanded) * 100.0 if commanded else 0.0,
        correction_factor=commanded / measured,
    )


def calculate_new_rotation_distance(
    old_rotation_distance: float,
    commanded: float,
    measured: float,
) -> float:
    """Compute the corrected ``rotation_distance``.

    Formula::

        new = old * (commanded / measured)

    Parameters
    ----------
    old_rotation_distance : float
        Current value from config (mm/rev).
    commanded : float
        Distance commanded (mm).
    measured : float
        Distance actually moved (mm).

    Returns
    -------
    float
        New ``rotation_distance`` value.
    """
    if measured <= 0:
        raise ValueError(f"measured must be > 0, got {measured}")
    return old_rotation_distance * (commanded / measured)


def calculate_tool_offset(
    pen_x: float,
    pen_y: float,
    airbrush_x: float,
    airbrush_y: float,
) -> tuple[float, float]:
    """Compute XY tool offset from crosshair measurements.

    Both tools drew a crosshair at the same commanded position.  The
    user measures the physical offset between the two marks.

    Parameters
    ----------
    pen_x, pen_y : float
        Measured centre of the pen crosshair (mm).
    airbrush_x, airbrush_y : float
        Measured centre of the airbrush crosshair (mm).

    Returns
    -------
    tuple[float, float]
        ``(offset_x, offset_y)`` to add to airbrush config so that it
        aligns with the pen.
    """
    return (pen_x - airbrush_x, pen_y - airbrush_y)


def binary_search_z(
    prompt_fn: callable,
    z_min: float,
    z_max: float,
    tolerance: float = 0.05,
) -> float:
    """Binary search for the Z touch-point.

    Parameters
    ----------
    prompt_fn : callable
        ``prompt_fn(z) -> bool`` -- move to Z and ask user "is the tool
        touching?" Returns ``True`` if touching.
    z_min, z_max : float
        Search bounds (mm).
    tolerance : float
        Stop when the search interval is smaller than this (mm).

    Returns
    -------
    float
        Z position at the touch boundary (mm).
    """
    while (z_max - z_min) > tolerance:
        mid = (z_min + z_max) / 2.0
        if prompt_fn(mid):
            # Touching -- move away (increase Z if pen, decrease if airbrush)
            z_min = mid
        else:
            z_max = mid
    return (z_min + z_max) / 2.0


def format_calibration_summary(results: dict[str, object]) -> str:
    """Format a calibration results dict as a human-readable summary.

    Parameters
    ----------
    results : dict
        Key-value pairs to display.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    lines = ["=" * 50, "  CALIBRATION RESULTS", "=" * 50]
    for key, val in results.items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
    lines.append("=" * 50)
    return "\n".join(lines)
