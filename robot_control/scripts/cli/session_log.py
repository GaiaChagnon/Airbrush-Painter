"""Session logging for robot CLI actions.

Writes timestamped, human-readable log lines to a file under
``robot_control/logs/``.  At session start the user is prompted to
overwrite the previous log or create a new one.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import questionary

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_DEFAULT_LOG = _LOGS_DIR / "session.log"


class SessionLog:
    """Append-only action log for a single CLI session.

    Parameters
    ----------
    log_dir : Path
        Directory where log files are written.
    """

    def __init__(self, log_dir: Path = _LOGS_DIR) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path: Path | None = None
        self._fh: Any = None
        self._start_time = time.monotonic()

    @property
    def path(self) -> Path | None:
        return self._path

    def prompt_and_open(self) -> None:
        """Interactively choose the log file and open it for writing.

        If ``session.log`` already exists the user is asked whether to
        overwrite (default) or create a timestamped file.
        """
        if _DEFAULT_LOG.exists() and _DEFAULT_LOG.stat().st_size > 0:
            overwrite = questionary.confirm(
                "Session log already exists. Overwrite?",
                default=True,
            ).ask()
            if overwrite is None:
                raise KeyboardInterrupt
            if overwrite:
                self._path = _DEFAULT_LOG
            else:
                stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                self._path = self._log_dir / f"session_{stamp}.log"
        else:
            self._path = _DEFAULT_LOG

        self._fh = open(self._path, "w", encoding="utf-8")  # noqa: SIM115
        self._write_header()
        logger.info("Session log: %s", self._path)

    def _write_header(self) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        self._write(f"# Robot CLI session started {ts}\n")

    def _write(self, line: str) -> None:
        if self._fh is not None:
            self._fh.write(line)
            self._fh.flush()

    def _timestamp(self) -> str:
        elapsed = time.monotonic() - self._start_time
        m, s = divmod(elapsed, 60)
        return f"[{int(m):02d}:{s:05.2f}]"

    def log_action(self, mode: str, action: str, details: str = "") -> None:
        """Log a user-visible action.

        Parameters
        ----------
        mode : str
            Active mode name (e.g. ``"interactive"``, ``"pump"``).
        action : str
            Short action label (e.g. ``"home"``, ``"dispense"``).
        details : str
            Free-form extra info.
        """
        parts = [self._timestamp(), f"[{mode}]", action]
        if details:
            parts.append(f"-- {details}")
        self._write(" ".join(parts) + "\n")

    def log_gcode(self, cmd: str) -> None:
        """Log a G-code command at debug level."""
        self._write(f"{self._timestamp()} [gcode] {cmd.strip()}\n")

    def log_error(self, exc: BaseException) -> None:
        """Log an exception."""
        self._write(f"{self._timestamp()} [ERROR] {type(exc).__name__}: {exc}\n")

    def close(self) -> None:
        """Flush and close the log file."""
        if self._fh is not None:
            ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            self._write(f"# Session ended {ts}\n")
            self._fh.close()
            self._fh = None
