"""Shared matplotlib helpers.

matplotlib is an optional runtime dependency; these helpers centralise the
lazy import and headless-backend setup so every plotting site behaves the same
and degrades gracefully when matplotlib is missing.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_pyplot():
    """Return a headless ``pyplot`` module, or ``None`` if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")
        return None


def prepare_output(path: str | Path) -> Path:
    """Ensure the parent directory of ``path`` exists and return it as a Path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
