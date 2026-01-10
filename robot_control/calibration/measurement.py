"""
Measurement Input and Calculations.

Helpers for interactive calibration routines that require user measurement input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MeasurementResult:
    """Result of a calibration measurement."""

    commanded: float
    measured: float
    error: float
    error_percent: float
    correction_factor: float


def get_float_input(prompt: str, default: float | None = None) -> float:
    """
    Prompt user for a floating point value.

    Parameters
    ----------
    prompt : str
        Prompt message to display.
    default : float | None
        Default value if user presses Enter.

    Returns
    -------
    float
        User-entered value.
    """
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        try:
            response = input(prompt).strip()
            if not response and default is not None:
                return default
            return float(response)
        except ValueError:
            print("Invalid number. Please enter a decimal value.")


def get_yes_no(prompt: str, default: bool | None = None) -> bool:
    """
    Prompt user for yes/no response.

    Parameters
    ----------
    prompt : str
        Prompt message to display.
    default : bool | None
        Default if user presses Enter. None means no default.

    Returns
    -------
    bool
        True for yes, False for no.
    """
    if default is True:
        prompt = f"{prompt} [Y/n]: "
    elif default is False:
        prompt = f"{prompt} [y/N]: "
    else:
        prompt = f"{prompt} [y/n]: "

    while True:
        response = input(prompt).strip().lower()
        if not response and default is not None:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def get_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """
    Prompt user to choose from options.

    Parameters
    ----------
    prompt : str
        Prompt message.
    options : list[str]
        List of option descriptions.
    default : int
        Default option index.

    Returns
    -------
    int
        Selected option index.
    """
    print(prompt)
    for i, option in enumerate(options):
        marker = "*" if i == default else " "
        print(f"  {marker} {i + 1}. {option}")

    while True:
        response = input(f"Choice [1-{len(options)}]: ").strip()
        if not response:
            return default
        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


def calculate_steps_correction(
    commanded: float,
    measured: float,
) -> MeasurementResult:
    """
    Calculate correction factor for steps/mm calibration.

    Parameters
    ----------
    commanded : float
        Distance that was commanded (mm).
    measured : float
        Distance that was actually traveled (mm).

    Returns
    -------
    MeasurementResult
        Calculated correction values.
    """
    if measured <= 0:
        raise ValueError("Measured distance must be positive")
    if commanded <= 0:
        raise ValueError("Commanded distance must be positive")

    error = commanded - measured
    error_percent = 100.0 * error / commanded
    correction_factor = commanded / measured

    return MeasurementResult(
        commanded=commanded,
        measured=measured,
        error=error,
        error_percent=error_percent,
        correction_factor=correction_factor,
    )


def calculate_new_rotation_distance(
    current_rotation_distance: float,
    correction_factor: float,
) -> float:
    """
    Calculate new rotation_distance for Klipper config.

    Parameters
    ----------
    current_rotation_distance : float
        Current Klipper rotation_distance value.
    correction_factor : float
        Calculated correction factor.

    Returns
    -------
    float
        New rotation_distance value.
    """
    # rotation_distance = (belt_pitch * pulley_teeth) / microsteps_per_step
    # If we're traveling too far, we need to decrease rotation_distance
    # If we're traveling too short, we need to increase rotation_distance
    return current_rotation_distance / correction_factor


def calculate_tool_offset(
    reference_pos: tuple[float, float],
    actual_pos: tuple[float, float],
) -> tuple[float, float]:
    """
    Calculate tool offset from crosshair alignment.

    Parameters
    ----------
    reference_pos : tuple[float, float]
        Position where reference tool drew crosshair.
    actual_pos : tuple[float, float]
        Position where second tool's crosshair actually appears.

    Returns
    -------
    tuple[float, float]
        XY offset to apply to second tool.
    """
    offset_x = reference_pos[0] - actual_pos[0]
    offset_y = reference_pos[1] - actual_pos[1]
    return (offset_x, offset_y)


def binary_search_z(
    initial_z: float,
    min_z: float,
    max_z: float,
    tolerance: float = 0.05,
) -> float | None:
    """
    Interactive binary search to find Z touch point.

    Parameters
    ----------
    initial_z : float
        Starting Z position.
    min_z : float
        Minimum allowed Z.
    max_z : float
        Maximum allowed Z.
    tolerance : float
        Stop when range is smaller than this.

    Returns
    -------
    float | None
        Found Z position, or None if cancelled.
    """
    low = min_z
    high = max_z
    current = initial_z

    print(f"\nBinary search for Z touch point (tolerance: {tolerance}mm)")
    print("Answer 'y' if tool is touching, 'n' if not touching, 'q' to quit")

    while (high - low) > tolerance:
        current = (low + high) / 2
        print(f"\nCurrent Z: {current:.3f}mm")

        response = input("Is tool touching? [y/n/q]: ").strip().lower()

        if response == "q":
            return None
        elif response == "y":
            # Touching - need to go higher (less pressure)
            low = current
        elif response == "n":
            # Not touching - need to go lower (more contact)
            high = current
        else:
            print("Please enter 'y', 'n', or 'q'")
            continue

    final = (low + high) / 2
    print(f"\nFound Z touch point: {final:.3f}mm")
    return final


def format_calibration_summary(results: dict) -> str:
    """
    Format calibration results for display.

    Parameters
    ----------
    results : dict
        Calibration results dictionary.

    Returns
    -------
    str
        Formatted summary text.
    """
    lines = [
        "=" * 50,
        "CALIBRATION RESULTS",
        "=" * 50,
    ]

    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        elif isinstance(value, tuple) and len(value) == 2:
            lines.append(f"  {key}: ({value[0]:.3f}, {value[1]:.3f})")
        else:
            lines.append(f"  {key}: {value}")

    lines.append("=" * 50)
    return "\n".join(lines)
