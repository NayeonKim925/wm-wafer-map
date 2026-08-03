"""Device selection utilities."""

from __future__ import annotations

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference to a concrete :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU.  An explicit request for
    an unavailable accelerator falls back to CPU with a warning rather than
    crashing, so the pipeline always runs somewhere.
    """
    preference = (preference or "auto").lower()

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif preference == "mps":
        if not _mps_available():
            logger.warning("MPS requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    return device


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())
