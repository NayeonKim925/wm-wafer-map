"""Centralised logging configuration.

Library code never calls :func:`print`; it obtains a module logger via
:func:`get_logger` and logs through it.  Applications (the entry-point scripts)
call :func:`setup_logging` exactly once to attach a console handler and,
optionally, a file handler.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger with a console handler (and optional file).

    Idempotent: calling it repeatedly will not stack duplicate handlers unless
    ``force`` is set (used to redirect logs to a fresh run directory).

    Returns the root logger.
    """
    global _CONFIGURED

    root = logging.getLogger()
    numeric_level = _coerce_level(level)
    root.setLevel(numeric_level)

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        _CONFIGURED = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if not _CONFIGURED:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        root.addHandler(console)
        _CONFIGURED = True

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid attaching two handlers to the same file.
        existing = {
            Path(h.baseFilename).resolve()
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if path.resolve() not in existing:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger (defaults to the package logger)."""
    return logging.getLogger(name if name else "wm811k")


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    if isinstance(resolved, int):
        return resolved
    return logging.INFO
