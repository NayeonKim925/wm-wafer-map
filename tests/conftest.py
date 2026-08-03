"""Shared pytest fixtures.

A small synthetic dataset with the exact WM-811K schema is generated on the fly
so the full pipeline can be tested without the real ~2 GB download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.labels import DEFECT_CLASSES, NONE_CLASS


def _make_wafer(pattern: str, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(20, 40))
    w = int(rng.integers(20, 40))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(r.max()) or 1.0
    wafer = np.zeros((h, w), dtype=np.int8)
    wafer[r <= rmax * 0.95] = 1
    if pattern == "Center":
        wafer[r <= rmax * 0.3] = 2
    elif pattern == "Donut":
        wafer[(r > rmax * 0.3) & (r < rmax * 0.55)] = 2
    elif pattern == "Edge-Ring":
        wafer[(r > rmax * 0.8) & (r <= rmax * 0.95)] = 2
    elif pattern == "Edge-Loc":
        wafer[(r > rmax * 0.8) & (xx > cx)] = 2
    elif pattern == "Loc":
        wafer[(yy > cy) & (xx > cx) & (r < rmax * 0.6)] = 2
    elif pattern == "Near-full":
        wafer[r <= rmax * 0.9] = 2
    elif pattern == "Scratch":
        wafer[np.abs(yy - xx) < 2] = 2
    elif pattern == "Random":
        wafer[(rng.random((h, w)) < 0.15) & (wafer == 1)] = 2
    return wafer


def _label_cell(name: str) -> np.ndarray:
    return np.array([[name]], dtype=object)


def build_synthetic_frame(
    n_per_class: int = 12, n_unlabelled: int = 6, n_lots: int = 15, seed: int = 0
) -> pd.DataFrame:
    """Build a DataFrame mirroring the WM-811K pickle schema.

    Includes multiple wafers per lot and a mix of official Training/Test flags
    so lot-aware and official split strategies can be exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in list(DEFECT_CLASSES) + [NONE_CLASS]:
        for _ in range(n_per_class):
            wafer = _make_wafer("none" if name == NONE_CLASS else name, rng)
            if name == NONE_CLASS:
                wafer[wafer == 2] = 1
            official = "Test" if rng.random() < 0.2 else "Training"
            rows.append(
                {
                    "waferMap": wafer,
                    "dieSize": float(wafer.size),
                    "lotName": f"lot{int(rng.integers(0, n_lots))}",
                    "waferIndex": float(rng.integers(1, 25)),
                    "trianTestLabel": _label_cell(official),
                    "failureType": _label_cell(name),
                }
            )
    for _ in range(n_unlabelled):
        wafer = _make_wafer("Random", rng)
        rows.append(
            {
                "waferMap": wafer,
                "dieSize": float(wafer.size),
                "lotName": f"lot{int(rng.integers(0, n_lots))}",
                "waferIndex": 1.0,
                "trianTestLabel": np.array([[]], dtype=object),
                "failureType": np.array([[]], dtype=object),
            }
        )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return build_synthetic_frame()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A datasets/wm811k directory containing a synthetic LSWMD.pkl."""
    directory = tmp_path / "datasets" / "wm811k"
    directory.mkdir(parents=True)
    build_synthetic_frame().to_pickle(directory / "LSWMD.pkl")
    return directory


@pytest.fixture
def make_config(tmp_path: Path):
    """Factory returning a small, fast Config pointed at temp locations."""

    def _factory(dataset_root: Path, **overrides: str) -> Config:
        base = [
            f"dataset.root_dir={dataset_root}",
            "dataset.min_file_size_mb=0",
            f"output.root_dir={tmp_path / 'outputs'}",
            "data.num_workers=0",
            "data.image_size=24",
            "data.batch_size=16",
            "model.base_channels=8",
            "training.epochs=1",
            "training.mixed_precision=false",
            "training.log_interval=100",
        ]
        base.extend(f"{key.replace('__', '.')}={value}" for key, value in overrides.items())
        return Config.load(overrides=base)

    return _factory
