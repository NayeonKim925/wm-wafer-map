"""Tests for the data loading / datamodule pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import WaferDataModule, load_wm811k
from src.data.labels import LabelMapping


def test_load_wm811k_drops_unlabelled(dataset_dir: Path):
    mapping = LabelMapping.build(include_none=True)
    wafers, labels = load_wm811k(dataset_dir / "LSWMD.pkl", mapping, include_none=True)
    assert len(wafers) == len(labels)
    # 9 classes * 12 per class = 108 labelled; unlabelled rows are dropped.
    assert len(labels) == 108
    assert set(np.unique(labels)).issubset(set(range(mapping.num_classes)))


def test_load_wm811k_exclude_none(dataset_dir: Path):
    mapping = LabelMapping.build(include_none=False)
    _, labels = load_wm811k(dataset_dir / "LSWMD.pkl", mapping, include_none=False)
    assert len(labels) == 96  # 8 defect classes * 12


def test_load_wm811k_missing_file(tmp_path: Path):
    mapping = LabelMapping.build()
    with pytest.raises(FileNotFoundError):
        load_wm811k(tmp_path / "nope.pkl", mapping)


def test_datamodule_splits_and_batches(make_config, dataset_dir: Path):
    cfg = make_config(dataset_dir.parent)
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()

    assert dm.num_classes == 9
    assert dm.input_channels == 3

    total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert total == 108
    # No overlap between splits.
    idx = np.concatenate(
        [dm.train_dataset.indices, dm.val_dataset.indices, dm.test_dataset.indices]
    )
    assert len(np.unique(idx)) == len(idx)

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1:] == (3, 24, 24)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_class_weights_normalised(make_config, dataset_dir: Path):
    cfg = make_config(dataset_dir.parent)
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
    weights = dm.class_weights()
    assert weights.shape == (9,)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)


def test_class_weights_before_setup_raises(make_config, dataset_dir: Path):
    cfg = make_config(dataset_dir.parent)
    dm = WaferDataModule(cfg, dataset_dir)
    with pytest.raises(RuntimeError):
        dm.class_weights()
