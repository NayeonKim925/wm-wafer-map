"""WM-811K label definitions and index mapping.

The dataset labels wafer maps with one of eight defect patterns or ``none``
(defect-free).  The ordering here is fixed and deterministic so that class
indices are stable across runs, machines and checkpoints.
"""

from __future__ import annotations

# The eight canonical defect patterns, in a fixed order.
DEFECT_CLASSES: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
)

# The defect-free class.
NONE_CLASS = "none"

ALL_CLASSES: tuple[str, ...] = DEFECT_CLASSES + (NONE_CLASS,)

# Case-insensitive lookup to the canonical spelling.
_CANONICAL = {name.lower(): name for name in ALL_CLASSES}


def canonicalize_label(name: object) -> str | None:
    """Return the canonical class spelling for ``name`` or ``None`` if unknown.

    Accepts any object (pandas may pass ``NaN``/``None`` for unlabelled rows);
    anything that is not a recognised class name maps to ``None``.
    """
    if not isinstance(name, str):
        return None
    return _CANONICAL.get(name.strip().lower())


class LabelMapping:
    """Bidirectional mapping between class names and integer indices."""

    def __init__(self, classes: list[str] | tuple[str, ...]) -> None:
        if not classes:
            raise ValueError("LabelMapping requires at least one class.")
        self.classes: list[str] = list(classes)
        self._to_index = {name: idx for idx, name in enumerate(self.classes)}

    @classmethod
    def build(cls, include_none: bool = True) -> LabelMapping:
        """Build the standard mapping, optionally including the ``none`` class."""
        classes = list(DEFECT_CLASSES)
        if include_none:
            classes.append(NONE_CLASS)
        return cls(classes)

    @classmethod
    def from_list(cls, classes: list[str]) -> LabelMapping:
        """Rebuild a mapping from a persisted class list (e.g. a checkpoint)."""
        return cls(classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def index(self, name: str) -> int:
        return self._to_index[name]

    def name(self, index: int) -> str:
        return self.classes[index]

    def __contains__(self, name: object) -> bool:
        return name in self._to_index

    def to_list(self) -> list[str]:
        return list(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LabelMapping({self.classes})"
