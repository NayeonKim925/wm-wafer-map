"""PyTorch dataset and raw WM-811K loading/cleaning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.labels import LabelMapping, canonicalize_label
from src.data.preprocessing import encode_wafer_map, normalize_failure_type
from src.utils.logging import get_logger

logger = get_logger(__name__)

# WM-811K pickle column names (``trianTestLabel`` keeps the dataset's typo).
_WAFER_COL = "waferMap"
_LABEL_COL = "failureType"
_LOT_COL = "lotName"
_SPLIT_COL = "trianTestLabel"


@dataclass
class RawWaferData:
    """Cleaned, in-memory view of the labelled WM-811K subset.

    Everything is index-aligned: ``labels[i]``, ``lot_names[i]`` and
    ``official_split[i]`` all describe ``wafer_maps[i]``.
    """

    wafer_maps: list[np.ndarray]
    labels: np.ndarray  # int64 class indices
    lot_names: np.ndarray  # str, the lotName each wafer belongs to
    official_split: np.ndarray  # str, "Training" | "Test" | "" (from trianTestLabel)
    label_mapping: LabelMapping

    def __len__(self) -> int:
        return len(self.labels)


class WaferMapDataset(Dataset):
    """A split view over *pre-resized* wafer maps.

    Resizing (the expensive, deterministic step) is done once up front by the
    data module; this dataset only performs the cheap categorical encoding and,
    optionally, augmentation.  Splits share the underlying ``resized_maps`` and
    hold only their own indices, keeping memory flat.
    """

    def __init__(
        self,
        resized_maps: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        representation: str = "onehot",
        augment: bool = False,
    ) -> None:
        self.resized_maps = resized_maps
        self.labels = labels
        self.indices = np.asarray(indices)
        self.representation = representation
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        global_idx = int(self.indices[i])
        wafer = self.resized_maps[global_idx]
        if self.augment:
            wafer = augment_wafer_map(wafer)
        tensor = encode_wafer_map(wafer, self.representation)
        label = int(self.labels[global_idx])
        return torch.from_numpy(np.ascontiguousarray(tensor)), label


def augment_wafer_map(wafer: np.ndarray) -> np.ndarray:
    """Label-preserving augmentation of a 2-D wafer grid.

    Wafer-defect classes are invariant to reflection and 90-degree rotation
    (a Center defect is still Center when flipped, etc.).  Uses the global
    NumPy RNG, which is seeded per DataLoader worker via :func:`seed_worker`.
    """
    if np.random.random() < 0.5:
        wafer = wafer[::-1, :]
    if np.random.random() < 0.5:
        wafer = wafer[:, ::-1]
    k = int(np.random.randint(0, 4))
    if k:
        wafer = np.rot90(wafer, k=k)
    return wafer


def seed_worker(_worker_id: int) -> None:
    """Per-worker RNG seeding (the canonical PyTorch reproducibility recipe).

    Without this, forked DataLoader workers share NumPy's RNG state and emit
    identical "random" augmentations.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_wm811k(
    pkl_path: str | Path,
    label_mapping: LabelMapping,
    *,
    include_none: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> RawWaferData:
    """Load and clean the WM-811K pickle into a :class:`RawWaferData`.

    Steps: read pickle -> canonicalise labels -> drop unlabelled/unknown rows
    -> optionally drop the ``none`` class -> optional stratified subsample. The
    lotName and official train/test flag are retained for leakage-free and
    benchmark-comparable splitting downstream.
    """
    path = Path(pkl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset pickle not found: {path}")

    logger.info("Loading WM-811K pickle from %s ...", path)
    frame = pd.read_pickle(path)
    logger.info("Loaded %d rows.", len(frame))

    for column in (_WAFER_COL, _LABEL_COL):
        if column not in frame.columns:
            raise KeyError(
                f"Expected column {column!r} not found in the dataset. "
                f"Available columns: {list(frame.columns)}"
            )

    work = pd.DataFrame({"_wafer": frame[_WAFER_COL]})
    work["label"] = frame[_LABEL_COL].map(normalize_failure_type).map(canonicalize_label)
    work["lot"] = _extract_lots(frame)
    work["split"] = _extract_official_split(frame)

    work = work.dropna(subset=["label"])
    if not include_none:
        work = work[work["label"] != "none"]
    work = work[work["label"].isin(set(label_mapping.classes))]

    if work.empty:
        raise ValueError(
            "No labelled wafer maps remain after cleaning. Check include_none and "
            "that the dataset pickle is the expected WM-811K file."
        )

    if max_samples is not None and len(work) > max_samples:
        keep = _stratified_subsample_indices(work["label"].to_numpy(), max_samples, seed)
        work = work.iloc[keep]
        logger.info("Subsampled to %d rows (max_samples=%d).", len(work), max_samples)

    labels = np.asarray([label_mapping.index(name) for name in work["label"]], dtype=np.int64)
    raw = RawWaferData(
        wafer_maps=[np.asarray(w) for w in work["_wafer"].tolist()],
        labels=labels,
        lot_names=work["lot"].to_numpy().astype(str),
        official_split=work["split"].to_numpy().astype(str),
        label_mapping=label_mapping,
    )
    _log_class_distribution(labels, label_mapping)
    logger.info("Distinct lots: %d", len(np.unique(raw.lot_names)))
    return raw


def _extract_lots(frame: pd.DataFrame) -> np.ndarray:
    """Return a lotName per row, or synthetic per-row lots if the column is absent."""
    if _LOT_COL in frame.columns:
        return frame[_LOT_COL].map(lambda v: str(v)).to_numpy()
    logger.warning(
        "Column %r not found; treating every wafer as its own lot "
        "(lot-aware splitting degrades to random).",
        _LOT_COL,
    )
    return np.array([f"__row_{i}" for i in range(len(frame))], dtype=object)


def _extract_official_split(frame: pd.DataFrame) -> np.ndarray:
    """Return the official Training/Test flag per row (empty string if absent)."""
    if _SPLIT_COL not in frame.columns:
        return np.array([""] * len(frame), dtype=object)
    return frame[_SPLIT_COL].map(normalize_failure_type).map(lambda v: v or "").to_numpy()


def _stratified_subsample_indices(
    labels: np.ndarray, max_samples: int, seed: int
) -> np.ndarray:
    """Positional indices for a class-stratified subsample of ``labels``."""
    rng = np.random.default_rng(seed)
    fraction = max_samples / len(labels)
    kept: list[np.ndarray] = []
    for value in np.unique(labels):
        idx = np.where(labels == value)[0]
        take = min(len(idx), max(1, int(round(len(idx) * fraction))))
        kept.append(rng.choice(idx, size=take, replace=False))
    keep = np.concatenate(kept)
    if len(keep) > max_samples:
        keep = rng.choice(keep, size=max_samples, replace=False)
    return np.sort(keep)


def _log_class_distribution(labels: np.ndarray, label_mapping: LabelMapping) -> None:
    counts = np.bincount(labels, minlength=label_mapping.num_classes)
    distribution = ", ".join(
        f"{label_mapping.name(i)}={int(count)}" for i, count in enumerate(counts)
    )
    logger.info("Class distribution: %s", distribution)
