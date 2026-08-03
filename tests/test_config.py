"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config, ConfigError


def test_defaults_load():
    cfg = Config.load()
    assert cfg.seed == 42
    assert cfg.dataset.kaggle_handle == "qingyi/wm811k-wafer-map"
    assert isinstance(cfg.dataset.root_dir, Path)


def test_overrides_are_typed():
    cfg = Config.load(overrides=["training.epochs=7", "data.batch_size=64", "training.mixed_precision=false"])
    assert cfg.training.epochs == 7
    assert isinstance(cfg.training.epochs, int)
    assert cfg.data.batch_size == 64
    assert cfg.training.mixed_precision is False


def test_nested_override_creates_path():
    cfg = Config.load(overrides=["training.optimizer.lr=0.005"])
    assert cfg.training.optimizer.lr == pytest.approx(0.005)


def test_unknown_key_raises():
    with pytest.raises(ConfigError):
        Config.load(overrides=["training.does_not_exist=1"])


def test_missing_config_file_raises():
    with pytest.raises(ConfigError):
        Config.load(config_path="nope/not-here.yaml")


def test_malformed_override_raises():
    with pytest.raises(ConfigError):
        Config.load(overrides=["this-has-no-equals"])


def test_roundtrip_save_and_from_dict(tmp_path: Path):
    cfg = Config.load(overrides=["model.name=wafer_resnet", "training.epochs=3"])
    out = cfg.save(tmp_path / "cfg.yaml")
    assert out.is_file()

    rebuilt = Config.from_dict(cfg.to_dict())
    assert rebuilt.model.name == "wafer_resnet"
    assert rebuilt.training.epochs == 3
    assert isinstance(rebuilt.output.root_dir, Path)


def test_from_dict_with_overrides():
    cfg = Config.from_dict({"seed": 1}, overrides=["seed=99"])
    assert cfg.seed == 99
