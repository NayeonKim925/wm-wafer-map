"""End-to-end tests driving the actual CLI entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.inference import load_wafer_maps
from src.utils.paths import get_project_root


def _load_entry_module(name: str):
    """Import a root-level entry script (train.py / evaluate.py) as a module."""
    path = get_project_root() / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_entry_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_main_end_to_end(dataset_dir: Path, tmp_path: Path):
    train = _load_entry_module("train")
    outputs = tmp_path / "outputs"
    argv = [
        "--set",
        f"dataset.root_dir={dataset_dir.parent}",
        "dataset.min_file_size_mb=0",
        f"output.root_dir={outputs}",
        "output.timestamp_subdir=false",
        "data.num_workers=0",
        "data.image_size=24",
        "data.batch_size=16",
        "model.base_channels=8",
        "training.epochs=1",
        "training.mixed_precision=false",
    ]
    assert train.main(argv) == 0

    run_dir = outputs / "wafer_cnn"
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "label_mapping.json").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "evaluation" / "classification_report.json").is_file()


def test_train_main_bad_config_returns_nonzero(tmp_path: Path):
    train = _load_entry_module("train")
    assert train.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_load_wafer_maps_npy(tmp_path: Path, synthetic_frame):
    wafers = synthetic_frame["waferMap"].tolist()[:4]
    path = tmp_path / "w.npy"
    np.save(path, np.array(wafers, dtype=object), allow_pickle=True)
    loaded = load_wafer_maps(path)
    assert len(loaded) == 4


def test_load_wafer_maps_single_npy(tmp_path: Path, synthetic_frame):
    wafer = np.asarray(synthetic_frame["waferMap"].iloc[0])
    path = tmp_path / "single.npy"
    np.save(path, wafer)
    loaded = load_wafer_maps(path)
    assert len(loaded) == 1
    assert loaded[0].ndim == 2


def test_load_wafer_maps_pickle_dataframe(dataset_dir: Path):
    loaded = load_wafer_maps(dataset_dir / "LSWMD.pkl")
    assert len(loaded) > 0


def test_load_wafer_maps_unsupported(tmp_path: Path):
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError):
        load_wafer_maps(bad)
