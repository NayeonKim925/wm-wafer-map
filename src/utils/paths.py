"""Path helpers.

Centralises project-root discovery so that relative paths configured in YAML
always resolve to the same location regardless of the current working
directory.  Nothing in the codebase hardcodes an absolute path; every path is
either configurable or derived from :func:`get_project_root`.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at <root>/src/utils/paths.py, so the repository root is three
# levels up.  Resolving here keeps the anchor independent of the caller's CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the absolute path to the repository root."""
    return _PROJECT_ROOT


def resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve ``path`` to an absolute :class:`~pathlib.Path`.

    Absolute paths are returned unchanged; relative paths are interpreted
    relative to the project root (or an explicit ``root`` if provided).
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base = root if root is not None else get_project_root()
    return (base / p).resolve()
