"""Wafer-map preprocessing.

WM-811K wafer maps are variable-sized 2-D integer grids where each cell is one
of ``{0: outside wafer / background, 1: passing die, 2: failing die}``.  These
helpers turn a raw grid into a fixed-size, model-ready ``(C, H, W)`` float
tensor, and normalise the dataset's quirky nested-array labels into clean
strings.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

# Supported encodings and their channel counts.
_CHANNELS = {"scalar": 1, "onehot": 3}


def num_input_channels(representation: str) -> int:
    """Number of tensor channels produced for a given representation."""
    try:
        return _CHANNELS[representation]
    except KeyError as exc:
        raise ValueError(
            f"Unknown representation {representation!r}; expected one of {sorted(_CHANNELS)}."
        ) from exc


def normalize_failure_type(value: Any) -> str | None:
    """Extract a scalar label string from WM-811K's nested-array cells.

    The ``failureType`` / ``trianTestLabel`` columns store values such as
    ``np.array([['Center']])`` for labelled rows and empty arrays (``[[]]``) for
    unlabelled rows.  Returns the inner string, or ``None`` when unlabelled.
    """
    current = value
    # Unwrap nested sequences/arrays down to a scalar.
    while isinstance(current, (list, tuple, np.ndarray)):
        if len(current) == 0:
            return None
        current = current[0]

    if current is None:
        return None
    if isinstance(current, float) and math.isnan(current):
        return None

    text = str(current).strip()
    return text or None


def resize_wafer_map(wafer: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D wafer grid to ``size x size`` using nearest-neighbour.

    Nearest-neighbour interpolation preserves the categorical ``{0, 1, 2}``
    cell values (no fractional cells are introduced).
    """
    arr = np.asarray(wafer)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D wafer map, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Cannot resize an empty wafer map.")

    image = Image.fromarray(arr.astype(np.uint8))
    resized = image.resize((size, size), resample=Image.NEAREST)
    return np.asarray(resized)


def encode_wafer_map(resized: np.ndarray, representation: str = "onehot") -> np.ndarray:
    """Encode a resized wafer grid into a ``(C, H, W)`` float32 array."""
    arr = np.asarray(resized)

    if representation == "scalar":
        # {0, 1, 2} -> {0.0, 0.5, 1.0}, single channel.
        scaled = (arr.astype(np.float32) / 2.0)
        return scaled[np.newaxis, :, :]

    if representation == "onehot":
        height, width = arr.shape
        onehot = np.zeros((3, height, width), dtype=np.float32)
        for value in (0, 1, 2):
            onehot[value] = (arr == value)
        return onehot

    raise ValueError(
        f"Unknown representation {representation!r}; expected one of {sorted(_CHANNELS)}."
    )


def preprocess_wafer_map(
    wafer: np.ndarray,
    image_size: int,
    representation: str = "onehot",
) -> np.ndarray:
    """Full pipeline: resize then encode into a ``(C, H, W)`` float32 array."""
    resized = resize_wafer_map(wafer, image_size)
    return encode_wafer_map(resized, representation)
