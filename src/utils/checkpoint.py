"""Model checkpoint (de)serialisation.

A checkpoint is self-describing: alongside the model weights it stores the model
config, the label mapping and the run config, so that evaluation and inference
can rebuild an identical model from the file alone — no external state needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, Any],
    model_config: dict[str, Any],
    label_classes: list[str],
    epoch: int,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Persist a checkpoint, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model_state,
        "model_config": model_config,
        "label_classes": list(label_classes),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    torch.save(payload, out)
    logger.debug("Saved checkpoint to %s", out)
    return out


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict saved by :func:`save_checkpoint`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        payload = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not accept weights_only.
        payload = torch.load(p, map_location=map_location)
    logger.debug("Loaded checkpoint from %s", p)
    return payload
