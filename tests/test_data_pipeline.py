"""Tests for the data loading / datamodule pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import RawWaferData, WaferDataModule, load_wm811k
from src.data.labels import LabelMapping


def test_load_wm811k_structured_output(dataset_dir: Path):
    mapping = LabelMapping.build(include_none=True)
    raw = load_wm811k(dataset_dir / "LSWMD.pkl", mapping, include_none=True)
    assert isinstance(raw, RawWaferData)
    # 9 classes * 12 per class = 108 labelled; unlabelled rows are dropped.
    assert len(raw) == 108
    assert len(raw.wafer_maps) == len(raw.labels) == len(raw.lot_names) == 108
    assert set(np.unique(raw.labels)).issubset(set(range(mapping.num_classes)))
    assert set(np.unique(raw.official_split)).issubset({"Training", "Test", ""})


def test_load_wm811k_exclude_none(dataset_dir: Path):
    mapping = LabelMapping.build(include_none=False)
    raw = load_wm811k(dataset_dir / "LSWMD.pkl", mapping, include_none=False)
    assert len(raw) == 96  # 8 defect classes * 12


def test_load_wm811k_missing_file(tmp_path: Path):
    mapping = LabelMapping.build()
    with pytest.raises(FileNotFoundError):
        load_wm811k(tmp_path / "nope.pkl", mapping)


def test_load_wm811k_subsample(dataset_dir: Path):
    mapping = LabelMapping.build()
    raw = load_wm811k(dataset_dir / "LSWMD.pkl", mapping, max_samples=27)
    assert len(raw) <= 27
    # stratified subsample keeps every class represented
    assert len(np.unique(raw.labels)) == mapping.num_classes


@pytest.mark.parametrize("strategy", ["lot", "official", "random"])
def test_datamodule_splits_cover_dataset(make_config, dataset_dir: Path, strategy):
    cfg = make_config(dataset_dir.parent, data__split_strategy=strategy)
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()

    assert dm.num_classes == 9
    assert dm.input_channels == 3

    idx = np.concatenate(
        [dm.train_dataset.indices, dm.val_dataset.indices, dm.test_dataset.indices]
    )
    assert len(np.unique(idx)) == len(idx)  # no overlap
    assert len(idx) == 108  # full coverage
    assert len(dm.test_dataset) > 0


def test_lot_split_has_no_leakage(make_config, dataset_dir: Path):
    """The whole point of lot splitting: no lot spans two splits."""
    cfg = make_config(dataset_dir.parent, data__split_strategy="lot")
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()

    lots = dm._raw.lot_names
    train_lots = set(lots[dm.train_dataset.indices])
    val_lots = set(lots[dm.val_dataset.indices])
    test_lots = set(lots[dm.test_dataset.indices])
    assert train_lots.isdisjoint(test_lots)
    assert train_lots.isdisjoint(val_lots)
    assert val_lots.isdisjoint(test_lots)


def test_official_train_val_lot_disjoint(make_config, dataset_dir: Path):
    """For the official split, the validation carve-out is still lot-aware."""
    cfg = make_config(dataset_dir.parent, data__split_strategy="official")
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
    lots = dm._raw.lot_names
    train_lots = set(lots[dm.train_dataset.indices])
    val_lots = set(lots[dm.val_dataset.indices])
    assert train_lots.isdisjoint(val_lots)


def test_official_split_uses_test_rows(make_config, dataset_dir: Path):
    cfg = make_config(dataset_dir.parent, data__split_strategy="official")
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
    # Every test-split wafer must carry the official 'Test' flag.
    official = dm._raw.official_split[dm.test_dataset.indices]
    assert np.all(official == "Test")


def test_batch_shapes_and_dtypes(make_config, dataset_dir: Path):
    cfg = make_config(dataset_dir.parent)
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
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
