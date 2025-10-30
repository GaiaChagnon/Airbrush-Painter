"""Unified logging configuration for all entrypoints.

Provides consistent logging across train.py, paint.py, GUI, calibration:
    - Console and file handlers with rotation
    - JSON output mode for ingestion (ELK, Vector, etc.)
    - Contextual fields (app, trial, run_id, epoch)
    - Warning capture (Python warnings → logging)
    - Uncaught exception logging
    - Multi-process support (QueueHandler/QueueListener for HPO)

Public API:
    setup_logging(**yaml_cfg.logging, context={"app": "train"})
    get_logger(name)
    push_context(trial=17, epoch=120)
    pop_context(keys=["epoch"])
    install_excepthook()

Format examples:
    Human: 2025-10-28T13:45:12.345Z | INFO | app=train trial=17 | Message
    JSON: {"t":"2025-10-28T13:45:12.345Z","lvl":"INFO","trial":17,"msg":"..."}

Context uses contextvars for thread/process isolation.
Idempotent: repeated setup_logging() calls don't duplicate handlers.
"""

import contextvars
import json
import logging
import logging.handlers
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Context variable for per-thread/process contextual fields
_context_var = contextvars.ContextVar('logging_context', default={})

# Track if logging has been configured (idempotency)
_configured = False


class ContextFormatter(logging.Formatter):
    """Custom formatter that includes contextual fields.

    Supports:
        - Human-readable format with colors (optional)
        - JSON format for machine ingestion
        - Contextual fields from push_context()
    """

    def __init__(
        self,
        fmt_mode: str = "human",
        use_color: bool = True,
        tz: str = "UTC"
    ):
        super().__init__()
        self.fmt_mode = fmt_mode
        self.use_color = use_color and sys.stderr.isatty()
        self.tz = tz
        
        # ANSI color codes
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Get contextual fields
        context = _context_var.get({})
        
        # Timestamp
        if self.tz == "UTC":
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        else:
            ts = datetime.fromtimestamp(record.created)
        
        if self.fmt_mode == "json":
            return self._format_json(record, ts, context)
        else:
            return self._format_human(record, ts, context)

    def _format_json(
        self,
        record: logging.LogRecord,
        ts: datetime,
        context: dict
    ) -> str:
        """Format as JSON line."""
        log_dict = {
            't': ts.isoformat(),
            'lvl': record.levelname,
            'name': record.name,
            'pid': os.getpid(),
            'msg': record.getMessage()
        }
        
        # Add contextual fields
        log_dict.update(context)
        
        # Add exception info if present
        if record.exc_info:
            log_dict['exc'] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict)

    def _format_human(
        self,
        record: logging.LogRecord,
        ts: datetime,
        context: dict
    ) -> str:
        """Format as human-readable line."""
        # Timestamp
        ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        # Level with optional color
        level = record.levelname
        if self.use_color:
            level = f"{self.colors.get(level, '')}{level:8s}{self.colors['RESET']}"
        else:
            level = f"{level:8s}"
        
        # Contextual fields
        context_str = ' '.join(f"{k}={v}" for k, v in context.items())
        if context_str:
            context_str = f"{context_str} | "
        
        # Message
        msg = record.getMessage()
        
        # Assemble
        parts = [ts_str, '|', level, '|']
        if context_str:
            parts.append(context_str)
        parts.append(msg)
        
        line = ' '.join(parts)
        
        # Add exception if present
        if record.exc_info:
            line += '\n' + self.formatException(record.exc_info)
        
        return line


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    *,
    json: bool = False,
    color: bool = True,
    to_stderr: bool = True,
    rotate: Optional[Dict[str, Any]] = None,
    tz: str = "UTC",
    capture_warnings: bool = True,
    quiet_libs: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    queue: Optional[Any] = None
) -> Dict[str, Any]:
    """Configure root logger (idempotent).

    Parameters
    ----------
    log_level : str
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_file : str, optional
        Log file path; None for no file logging
    json : bool
        Use JSON format (for tooling ingestion), default False
    color : bool
        Use ANSI colors in console output, default True
    to_stderr : bool
        Log to stderr (console), default True
    rotate : dict, optional
        Rotation config:
        - {"mode": "size", "max_bytes": 50_000_000, "backup_count": 5}
        - {"mode": "time", "when": "D", "interval": 1, "backup_count": 7}
    tz : str
        Timezone for timestamps, "UTC" (default) or "local"
    capture_warnings : bool
        Capture Python warnings to logging, default True
    quiet_libs : list[str], optional
        Library names to set to WARNING level (e.g., ["matplotlib", "PIL"])
    context : dict, optional
        Initial contextual fields (e.g., {"app": "train"})
    queue : multiprocessing.Queue, optional
        Queue for multi-process logging (parent creates listener)

    Returns
    -------
    dict
        Configuration info: {"handlers": [...], "listener": optional}

    Notes
    -----
    Idempotent: repeated calls won't duplicate handlers.
    For multi-process: parent calls with queue and starts listener;
    children call with same queue (handlers use QueueHandler).

    Examples
    --------
    >>> setup_logging(log_level="INFO", log_file="outputs/logs/train.log",
    ...               rotate={"mode": "size", "max_bytes": 50_000_000, "backup_count": 5},
    ...               context={"app": "train"})
    """
    global _configured
    
    # Get root logger
    root = logging.getLogger()
    
    # If already configured, clear existing handlers
    if _configured:
        root.handlers.clear()
    
    # Set level
    root.setLevel(getattr(logging, log_level.upper()))
    
    handlers = []
    listener = None
    
    # Multi-process mode: use QueueHandler
    if queue is not None:
        try:
            from logging.handlers import QueueHandler, QueueListener
            
            # Child processes use QueueHandler
            if to_stderr or log_file:
                # Parent: create listener with actual handlers
                actual_handlers = []
                
                if to_stderr:
                    console_handler = logging.StreamHandler(sys.stderr)
                    console_handler.setFormatter(ContextFormatter("human", color, tz))
                    actual_handlers.append(console_handler)
                
                if log_file:
                    file_handler = _create_file_handler(log_file, rotate, json, tz)
                    actual_handlers.append(file_handler)
                
                # Start listener
                listener = QueueListener(queue, *actual_handlers, respect_handler_level=True)
                listener.start()
            
            # All processes (parent and children) use QueueHandler
            queue_handler = QueueHandler(queue)
            root.addHandler(queue_handler)
            handlers.append(queue_handler)
        
        except ImportError:
            # Fallback if QueueHandler not available
            pass
    
    else:
        # Standard mode: direct handlers
        if to_stderr:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(ContextFormatter("human", color, tz))
            root.addHandler(console_handler)
            handlers.append(console_handler)
        
        if log_file:
            file_handler = _create_file_handler(log_file, rotate, json, tz)
            root.addHandler(file_handler)
            handlers.append(file_handler)
    
    # Set contextual fields
    if context:
        push_context(**context)
    
    # Quiet noisy libraries
    if quiet_libs:
        for lib in quiet_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Capture warnings
    if capture_warnings:
        route_warnings()
    
    _configured = True
    
    return {
        'handlers': handlers,
        'listener': listener
    }


def _create_file_handler(
    log_file: str,
    rotate: Optional[Dict[str, Any]],
    json_format: bool,
    tz: str
) -> logging.Handler:
    """Create file handler with optional rotation."""
    # Ensure directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rotate:
        mode = rotate.get('mode', 'size')
        
        if mode == 'size':
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=rotate.get('max_bytes', 50_000_000),
                backupCount=rotate.get('backup_count', 5)
            )
        elif mode == 'time':
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when=rotate.get('when', 'D'),
                interval=rotate.get('interval', 1),
                backupCount=rotate.get('backup_count', 7)
            )
        else:
            raise ValueError(f"Unknown rotation mode: {mode}. Use 'size' or 'time'.")
    else:
        handler = logging.FileHandler(log_file)
    
    # Set formatter
    fmt_mode = "json" if json_format else "human"
    handler.setFormatter(ContextFormatter(fmt_mode, use_color=False, tz=tz))
    
    return handler


def get_logger(name: str) -> logging.Logger:
    """Get logger by name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting training")
    """
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Update root logger level at runtime.

    Parameters
    ----------
    level : str
        New logging level: "DEBUG", "INFO", etc.

    Examples
    --------
    >>> set_level("DEBUG")  # Enable debug logging
    """
    logging.getLogger().setLevel(getattr(logging, level.upper()))


def push_context(**kwargs) -> None:
    """Add contextual fields to all subsequent log records.

    Parameters
    ----------
    **kwargs
        Key-value pairs to add (e.g., trial=17, epoch=120)

    Notes
    -----
    Context is thread/process-local (uses contextvars).
    Fields appear in all log messages until popped.

    Examples
    --------
    >>> push_context(app="train", trial=17)
    >>> logger.info("Started")  # → "... | app=train trial=17 | Started"
    >>> push_context(epoch=120)
    >>> logger.info("Checkpoint")  # → "... | app=train trial=17 epoch=120 | ..."
    """
    current = _context_var.get({})
    updated = {**current, **kwargs}
    _context_var.set(updated)


def pop_context(keys: Optional[List[str]] = None) -> None:
    """Remove contextual fields.

    Parameters
    ----------
    keys : list[str], optional
        Keys to remove; if None, clears all context

    Examples
    --------
    >>> pop_context(keys=["epoch"])  # Remove epoch, keep others
    >>> pop_context()  # Clear all context
    """
    if keys is None:
        _context_var.set({})
    else:
        current = _context_var.get({})
        for key in keys:
            current.pop(key, None)
        _context_var.set(current)


def install_excepthook() -> None:
    """Install handler to log uncaught exceptions.

    Notes
    -----
    Logs exception with traceback, then re-raises.
    Useful for debugging crashes in production.

    Examples
    --------
    >>> install_excepthook()
    >>> # Now uncaught exceptions are logged before program exits
    """
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger(__name__)
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = log_exception


def route_warnings() -> None:
    """Route Python warnings to logging.warning().

    Notes
    -----
    Called automatically if capture_warnings=True in setup_logging.
    """
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)


def shutdown() -> None:
    """Shutdown logging and flush handlers.

    Notes
    -----
    Call at end of main() to ensure all logs are written.
    Especially important for multi-process logging.
    """
    logging.shutdown()
