"""Shared input helpers for the robot CLI.

Thin wrappers around ``questionary.text()`` that parse, validate, and
retry numeric input.  These are extracted from the pump-controller
module so every CLI mode can reuse them without duplicating validation
logic.

All functions raise ``KeyboardInterrupt`` when the user cancels the
prompt (questionary returns ``None``).
"""

from __future__ import annotations

import logging

import questionary

logger = logging.getLogger(__name__)


def ask_float(
    prompt: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Prompt the user for a float, retrying on invalid or out-of-range input.

    Parameters
    ----------
    prompt : str
        Question text shown to the user.
    default : float
        Value returned when the user submits an empty response.
    min_val : float | None
        Inclusive lower bound.  ``None`` means no lower bound.
    max_val : float | None
        Inclusive upper bound.  ``None`` means no upper bound.

    Returns
    -------
    float
        The validated numeric value.

    Raises
    ------
    KeyboardInterrupt
        If the user cancels the prompt (Ctrl-C / Ctrl-D).
    """
    while True:
        result = questionary.text(
            f"{prompt} [{default:g}]:",
            default=str(default) if default else "",
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        val = result.strip()
        if not val:
            return default
        try:
            v = float(val)
        except ValueError:
            logger.warning("Invalid number: %r — please try again.", val)
            continue
        if min_val is not None and v < min_val:
            logger.warning(
                "Value %g is below minimum %g — please try again.",
                v,
                min_val,
            )
            continue
        if max_val is not None and v > max_val:
            logger.warning(
                "Value %g is above maximum %g — please try again.",
                v,
                max_val,
            )
            continue
        return v


def ask_int(
    prompt: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Prompt the user for an integer, retrying on invalid or out-of-range input.

    Parameters
    ----------
    prompt : str
        Question text shown to the user.
    default : int
        Value returned when the user submits an empty response.
    min_val : int | None
        Inclusive lower bound.  ``None`` means no lower bound.
    max_val : int | None
        Inclusive upper bound.  ``None`` means no upper bound.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    KeyboardInterrupt
        If the user cancels the prompt (Ctrl-C / Ctrl-D).
    """
    while True:
        result = questionary.text(
            f"{prompt} [{default}]:",
            default=str(default),
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        val = result.strip()
        if not val:
            return default
        try:
            v = int(val)
        except ValueError:
            logger.warning("Invalid integer: %r — please try again.", val)
            continue
        if min_val is not None and v < min_val:
            logger.warning(
                "Value %d is below minimum %d — please try again.",
                v,
                min_val,
            )
            continue
        if max_val is not None and v > max_val:
            logger.warning(
                "Value %d is above maximum %d — please try again.",
                v,
                max_val,
            )
            continue
        return v
