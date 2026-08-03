"""Model registry and factory.

New architectures are added by decorating a builder with ``@register_model``;
nothing else in the codebase needs to change.  ``build_model`` is the single
construction entry point used by training, evaluation and inference.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from src.config import ModelConfig
from src.models.cnn import WaferCNN
from src.models.resnet import WaferResNet
from src.utils.logging import get_logger

logger = get_logger(__name__)

# name -> builder(in_channels, num_classes, **kwargs) -> nn.Module
_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator registering a model builder under ``name``."""

    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model {name!r} is already registered.")
        _REGISTRY[key] = builder
        return builder

    return decorator


def available_models() -> list[str]:
    """Return the sorted list of registered architecture names."""
    return sorted(_REGISTRY)


def build_model(model_cfg: ModelConfig, in_channels: int, num_classes: int) -> nn.Module:
    """Instantiate the architecture named in ``model_cfg``.

    ``num_classes`` and ``in_channels`` are supplied by the data pipeline so the
    model always matches the data.
    """
    key = model_cfg.name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model {model_cfg.name!r}. Available: {available_models()}"
        )

    model = _REGISTRY[key](
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=model_cfg.base_channels,
        dropout=model_cfg.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built model '%s' (in_channels=%d, num_classes=%d, %.2fM trainable params).",
        model_cfg.name,
        in_channels,
        num_classes,
        n_params / 1e6,
    )
    return model


@register_model("wafer_cnn")
def _build_wafer_cnn(in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    return WaferCNN(in_channels=in_channels, num_classes=num_classes, **kwargs)


@register_model("wafer_resnet")
def _build_wafer_resnet(in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    return WaferResNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
