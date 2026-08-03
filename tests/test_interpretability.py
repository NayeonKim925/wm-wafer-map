"""Tests for Grad-CAM."""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.interpretability import GradCAM, find_target_layer
from src.models import build_model


def test_find_target_layer_returns_conv():
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=9)
    layer = find_target_layer(model)
    assert isinstance(layer, nn.Conv2d)


def test_gradcam_output_shape_and_range():
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=9)
    with GradCAM(model) as cam:
        x = torch.randn(2, 3, 32, 32)
        cams, indices = cam(x)
    assert cams.shape[0] == 2
    assert cams.ndim == 3
    assert cams.min() >= 0.0 and cams.max() <= 1.0 + 1e-6
    assert indices.shape == (2,)


def test_gradcam_overlay_matches_input_size():
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=9)
    cam = GradCAM(model)
    overlay = cam.overlay(torch.randn(1, 3, 40, 40), size=40)
    cam.remove()
    assert overlay.shape == (1, 40, 40)


def test_gradcam_hooks_are_removed():
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=9)
    layer = find_target_layer(model)
    before = len(layer._forward_hooks)
    with GradCAM(model):
        during = len(layer._forward_hooks)
    after = len(layer._forward_hooks)
    assert during == before + 1
    assert after == before


def test_gradcam_targets_specific_class():
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=9)
    with GradCAM(model) as cam:
        x = torch.randn(1, 3, 32, 32)
        _, indices = cam(x, class_indices=torch.tensor([3]))
    assert int(indices[0]) == 3


def test_no_conv_raises():
    import pytest

    with pytest.raises(ValueError):
        find_target_layer(nn.Sequential(nn.Linear(4, 4)))
