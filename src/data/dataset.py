"""PyTorch dataset and raw WM-811K loading/cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.labels import LabelMapping, canonicalize_label
from src.data.preprocessing import normalize_failure_type, preprocess_wafer_map
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Column names in the WM-811K pickle (``trianTestLabel`` keeps the dataset's
# original typo).
_WAFER_COL = "waferMap"
_LABEL_COL = "failureType"


class WaferMapDataset(Dataset):
    """A view over cleaned wafer maps for a single split.

    The wafer arrays and labels are shared (not copied) across the train/val/
    test splits; each dataset only holds the indices belonging to its split,
    keeping memory usage low.
    """

    def __init__(
        self,
        wafer_maps: list[np.ndarray],
        labels: np.ndarray,
        indices: np.ndarray,
        image_size: int,
        representation: str = "onehot",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.indices = np.asarray(indices)
        self.image_size = image_size
        self.representation = representation
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        global_idx = int(self.indices[i])
        wafer = self.wafer_maps[global_idx]
        label = int(self.labels[global_idx])

        tensor = preprocess_wafer_map(wafer, self.image_size, self.representation)
        if self.augment:
            tensor = self._augment(tensor)
        return torch.from_numpy(np.ascontiguousarray(tensor)), label

    def _augment(self, tensor: np.ndarray) -> np.ndarray:
        """Label-preserving flips / 90-degree rotations (H, W are the last axes)."""
        if self._rng.random() < 0.5:
            tensor = tensor[:, ::-1, :]
        if self._rng.random() < 0.5:
            tensor = tensor[:, :, ::-1]
        k = int(self._rng.integers(0, 4))
        if k:
            tensor = np.rot90(tensor, k=k, axes=(1, 2))
        return tensor


def load_wm811k(
    pkl_path: str | Path,
    label_mapping: LabelMapping,
    *,
    include_none: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load and clean the WM-811K pickle into wafer arrays and integer labels.

    Steps: read pickle -> canonicalise labels -> drop unlabelled/unknown rows
    -> optionally drop the ``none`` class -> optional stratified subsample.

    Returns
    -------
    (wafer_maps, labels)
        ``wafer_maps`` is a list of 2-D integer arrays; ``labels`` is an
        ``int64`` array of class indices aligned with the mapping.
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

    # Canonicalise labels; unlabelled/unknown rows become NaN and are dropped.
    frame = frame[[_WAFER_COL, _LABEL_COL]].copy()
    frame["label"] = (
        frame[_LABEL_COL].map(normalize_failure_type).map(canonicalize_label)
    )
    frame = frame.dropna(subset=["label"])

    if not include_none:
        frame = frame[frame["label"] != "none"]

    # Keep only classes present in the mapping (all, by construction).
    frame = frame[frame["label"].isin(set(label_mapping.classes))]

    if frame.empty:
        raise ValueError(
            "No labelled wafer maps remain after cleaning. Check include_none and "
            "that the dataset pickle is the expected WM-811K file."
        )

    if max_samples is not None and len(frame) > max_samples:
        frame = _stratified_subsample(frame, "label", max_samples, seed)
        logger.info("Subsampled to %d rows (max_samples=%d).", len(frame), max_samples)

    wafer_maps = [np.asarray(w) for w in frame[_WAFER_COL].tolist()]
    labels = np.asarray([label_mapping.index(name) for name in frame["label"]], dtype=np.int64)

    _log_class_distribution(labels, label_mapping)
    return wafer_maps, labels


def _stratified_subsample(
    frame: pd.DataFrame, label_col: str, max_samples: int, seed: int
) -> pd.DataFrame:
    """Subsample ``frame`` to roughly ``max_samples`` rows, keeping every class."""
    fraction = max_samples / len(frame)
    sampled = frame.groupby(label_col, group_keys=False).apply(
        lambda group: group.sample(
            n=max(1, int(round(len(group) * fraction))),
            random_state=seed,
        )
    )
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    return sampled.reset_index(drop=True)


def _log_class_distribution(labels: np.ndarray, label_mapping: LabelMapping) -> None:
    counts = np.bincount(labels, minlength=label_mapping.num_classes)
    distribution = ", ".join(
        f"{label_mapping.name(i)}={int(count)}" for i, count in enumerate(counts)
    )
    logger.info("Class distribution: %s", distribution)
