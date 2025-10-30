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

