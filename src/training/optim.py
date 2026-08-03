"""Optimizer and learning-rate scheduler construction from config."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from src.config import OptimizerConfig, SchedulerConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_optimizer(
    params: Iterable[nn.Parameter], cfg: OptimizerConfig
) -> torch.optim.Optimizer:
    """Create the optimizer named in ``cfg``."""
    name = cfg.name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.momentum > 0,
        )
    raise ValueError(f"Unknown optimizer {cfg.name!r}; expected adam | adamw | sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the LR scheduler named in ``cfg`` (``None`` disables scheduling)."""
    name = cfg.name.lower()
    if name in ("none", "off", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    raise ValueError(f"Unknown scheduler {cfg.name!r}; expected cosine | step | none.")
