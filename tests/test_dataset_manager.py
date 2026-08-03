"""Tests for DatasetManager: locate, download (mocked), verify, error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DatasetConfig
from src.data.dataset_manager import (
    DatasetDownloadError,
    DatasetManager,
    DatasetVerificationError,
)


def _config(root: Path, **kw) -> DatasetConfig:
    params = {
        "root_dir": root,
        "dataset_name": "wm811k",
        "pickle_filename": "LSWMD.pkl",
        "expected_files": ["LSWMD.pkl"],
        "min_file_size_mb": 0.0,
    }
    params.update(kw)
    return DatasetConfig(**params)


def test_verify_and_locate_with_existing_file(dataset_dir: Path):
    cfg = _config(dataset_dir.parent)  # root is <tmp>/datasets
    manager = DatasetManager(cfg)
    manager.verify()  # should not raise
    assert manager.is_available() is True
    assert manager.locate() == manager.dataset_dir


def test_verify_missing_file_raises(tmp_path: Path):
    cfg = _config(tmp_path / "datasets")
    manager = DatasetManager(cfg)
    with pytest.raises(DatasetVerificationError):
        manager.verify()
    assert manager.is_available() is False
    assert manager.locate() is None


def test_verify_undersized_file_raises(tmp_path: Path):
    root = tmp_path / "datasets"
    (root / "wm811k").mkdir(parents=True)
    (root / "wm811k" / "LSWMD.pkl").write_bytes(b"tiny")
    cfg = _config(root, min_file_size_mb=100.0)
    manager = DatasetManager(cfg)
    with pytest.raises(DatasetVerificationError):
        manager.verify()


def test_prepare_skips_when_present(dataset_dir: Path, monkeypatch):
    cfg = _config(dataset_dir.parent)
    manager = DatasetManager(cfg)

    def _boom(*_a, **_k):  # download must NOT be called
        raise AssertionError("download() should not run when data is present")

    monkeypatch.setattr(manager, "download", _boom)
    assert manager.prepare() == manager.dataset_dir


def test_prepare_downloads_when_missing(tmp_path: Path, monkeypatch, synthetic_frame):
    # A fake kagglehub cache directory containing the pickle.
    cache = tmp_path / "kaggle_cache"
    cache.mkdir()
    synthetic_frame.to_pickle(cache / "LSWMD.pkl")

    import kagglehub

    monkeypatch.setattr(kagglehub, "dataset_download", lambda *a, **k: str(cache))

    cfg = _config(tmp_path / "datasets", link_mode="copy")
    manager = DatasetManager(cfg)
    result = manager.prepare()

    assert result == manager.dataset_dir
    assert manager.primary_file.is_file()
    assert manager.is_available() is True


def test_download_finds_nested_file(tmp_path: Path, monkeypatch, synthetic_frame):
    cache = tmp_path / "kaggle_cache"
    (cache / "nested" / "deep").mkdir(parents=True)
    synthetic_frame.to_pickle(cache / "nested" / "deep" / "LSWMD.pkl")

    import kagglehub

    monkeypatch.setattr(kagglehub, "dataset_download", lambda *a, **k: str(cache))

    cfg = _config(tmp_path / "datasets", link_mode="symlink")
    manager = DatasetManager(cfg)
    manager.download()
    assert manager.primary_file.exists()


def test_download_missing_expected_file_raises(tmp_path: Path, monkeypatch):
    cache = tmp_path / "kaggle_cache"
    cache.mkdir()
    (cache / "OTHER.txt").write_text("not the pickle")

    import kagglehub

    monkeypatch.setattr(kagglehub, "dataset_download", lambda *a, **k: str(cache))

    cfg = _config(tmp_path / "datasets")
    manager = DatasetManager(cfg)
    with pytest.raises(DatasetDownloadError):
        manager.download()


def test_download_failure_gives_actionable_message(tmp_path: Path, monkeypatch):
    import kagglehub

    def _raise(*_a, **_k):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(kagglehub, "dataset_download", _raise)

    cfg = _config(tmp_path / "datasets")
    manager = DatasetManager(cfg)
    with pytest.raises(DatasetDownloadError) as exc_info:
        manager.download()
    message = str(exc_info.value)
    assert "KAGGLE_USERNAME" in message  # points the user at the fix


def test_describe_reports_state(dataset_dir: Path):
    cfg = _config(dataset_dir.parent)
    info = DatasetManager(cfg).describe()
    assert info["available"] is True
    assert info["files"][0]["name"] == "LSWMD.pkl"
    assert info["files"][0]["exists"] is True
