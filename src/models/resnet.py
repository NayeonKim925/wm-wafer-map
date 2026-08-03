"""A small ResNet-style classifier for wafer maps (an alternative architecture)."""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ResidualBlock


class WaferResNet(nn.Module):
    """Compact ResNet: a stem plus four residual stages with global pooling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        prev = widths[0]
        for i, width in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(ResidualBlock(prev, width, stride=stride))
            stages.append(ResidualBlock(width, width, stride=1))
            prev = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)
