"""Data module: turns a dataset directory into train/val/test dataloaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import WaferMapDataset, load_wm811k
from src.data.labels import LabelMapping
from src.data.preprocessing import num_input_channels
from src.utils.logging import get_logger


class WaferDataModule:
    """Prepare stratified dataloaders for the WM-811K classification task.

    Call :meth:`setup` once, then request loaders via :meth:`train_dataloader`,
    :meth:`val_dataloader` and :meth:`test_dataloader`.
    """

    def __init__(self, config: Config, dataset_dir: str | Path, logger=None) -> None:
        self.config = config
        self.data_cfg = config.data
        self.logger = logger or get_logger(__name__)

        self.dataset_dir = Path(dataset_dir)
        self.pkl_path = self.dataset_dir / config.dataset.pickle_filename
        self.label_mapping = LabelMapping.build(include_none=self.data_cfg.include_none)

        self._wafer_maps: list[np.ndarray] | None = None
        self._labels: np.ndarray | None = None
        self.train_dataset: WaferMapDataset | None = None
        self.val_dataset: WaferMapDataset | None = None
        self.test_dataset: WaferMapDataset | None = None
        self._train_indices: np.ndarray | None = None

    # -- Lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        """Load, clean and split the dataset into three dataset views."""
        wafer_maps, labels = load_wm811k(
            self.pkl_path,
            self.label_mapping,
            include_none=self.data_cfg.include_none,
            max_samples=self.data_cfg.max_samples,
            seed=self.config.seed,
        )
        self._wafer_maps = wafer_maps
        self._labels = labels

        train_idx, val_idx, test_idx = self._split(labels)
        self._train_indices = train_idx

        self.logger.info(
            "Split sizes -> train=%d, val=%d, test=%d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        common = dict(
            wafer_maps=wafer_maps,
            labels=labels,
            image_size=self.data_cfg.image_size,
            representation=self.data_cfg.representation,
            seed=self.config.seed,
        )
        self.train_dataset = WaferMapDataset(indices=train_idx, augment=True, **common)
        self.val_dataset = WaferMapDataset(indices=val_idx, augment=False, **common)
        self.test_dataset = WaferMapDataset(indices=test_idx, augment=False, **common)

    # -- Properties ---------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return self.label_mapping.num_classes

    @property
    def input_channels(self) -> int:
        return num_input_channels(self.data_cfg.representation)

    # -- Dataloaders --------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.train_dataset), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.val_dataset), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.test_dataset), shuffle=False)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from the *training* split.

        Weights are normalised to a mean of 1.0.  Classes absent from the
        training split receive a weight of 1.0.
        """
        if self._labels is None or self._train_indices is None:
            raise RuntimeError("Call setup() before requesting class weights.")

        counts = np.bincount(
            self._labels[self._train_indices], minlength=self.num_classes
        ).astype(np.float64)

        weights = np.ones(self.num_classes, dtype=np.float64)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    # -- Internals ----------------------------------------------------------
    def _split(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(labels))
        test_size = self.data_cfg.test_split
        val_size = self.data_cfg.val_split

        train_val_idx, test_idx = self._safe_split(indices, labels, test_size)

        # val_size is expressed as a fraction of the whole; rescale to the
        # train+val remainder.
        remaining = 1.0 - test_size
        val_relative = val_size / remaining if remaining > 0 else 0.0
        train_idx, val_idx = self._safe_split(
            train_val_idx, labels[train_val_idx], val_relative
        )
        return train_idx, val_idx, test_idx

    def _safe_split(
        self, indices: np.ndarray, labels: np.ndarray, test_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if test_size <= 0:
            return indices, np.array([], dtype=indices.dtype)
        try:
            left, right = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.config.seed,
                stratify=labels,
            )
        except ValueError:
            # Too few samples in some class to stratify; fall back gracefully.
            self.logger.warning(
                "Stratified split not possible (rare classes); using a random split."
            )
            left, right = train_test_split(
                indices, test_size=test_size, random_state=self.config.seed
            )
        return left, right

    def _make_loader(self, dataset: WaferMapDataset, shuffle: bool) -> DataLoader:
        pin_memory = self.data_cfg.pin_memory and torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.data_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _require(dataset: WaferMapDataset | None) -> WaferMapDataset:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return dataset
