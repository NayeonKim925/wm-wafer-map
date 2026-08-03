"""A compact, size-agnostic CNN classifier for wafer maps."""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock


class WaferCNN(nn.Module):
    """Four-stage convolutional classifier.

    Channel widths scale from ``base_channels`` (``[b, 2b, 4b, 8b]``).  A global
    average pool makes the network agnostic to the exact input resolution, so
    the same architecture works for 32x32 or 64x64 wafer maps.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        stages: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            stages.append(ConvBlock(prev, width))
            prev = width
        self.features = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
