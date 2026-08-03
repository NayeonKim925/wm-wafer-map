"""Tests for model construction and the registry."""

from __future__ import annotations

import pytest
import torch

from src.config import ModelConfig
from src.models import available_models, build_model
from src.models.factory import register_model


@pytest.mark.parametrize("name", ["wafer_cnn", "wafer_resnet"])
@pytest.mark.parametrize("channels", [1, 3])
def test_build_and_forward(name, channels):
    cfg = ModelConfig(name=name, base_channels=8)
    model = build_model(cfg, in_channels=channels, num_classes=9)
    out = model(torch.randn(2, channels, 32, 32))
    assert out.shape == (2, 9)


def test_size_agnostic():
    cfg = ModelConfig(name="wafer_cnn", base_channels=8)
    model = build_model(cfg, in_channels=3, num_classes=9)
    for size in (24, 32, 64):
        out = model(torch.randn(1, 3, size, size))
        assert out.shape == (1, 9)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="does_not_exist"), in_channels=3, num_classes=9)


def test_registry_lists_models():
    models = available_models()
    assert "wafer_cnn" in models
    assert "wafer_resnet" in models


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register_model("wafer_cnn")
        def _dupe(**_kw):  # pragma: no cover - registration raises first
            return None
