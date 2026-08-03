"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    ``deterministic`` additionally forces cuDNN into deterministic mode, which
    improves reproducibility at some cost to throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        logger.debug("NumPy not available while seeding.")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        logger.debug("PyTorch not available while seeding.")

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
