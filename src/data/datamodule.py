"""Data module: turns a dataset directory into train/val/test dataloaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import RawWaferData, WaferMapDataset, load_wm811k, seed_worker
from src.data.labels import LabelMapping
from src.data.preprocessing import num_input_channels, resize_wafer_stack
from src.utils.logging import get_logger

# A named tuple of positional index arrays for the three splits.
Split = tuple[np.ndarray, np.ndarray, np.ndarray]


class WaferDataModule:
    """Prepare train/val/test dataloaders for WM-811K classification.

    Call :meth:`setup` once, then request loaders. The split strategy
    (``lot`` / ``official`` / ``random``) is chosen in the config; ``lot`` and
    ``official`` are leakage-free, ``random`` is not.
    """

    def __init__(self, config: Config, dataset_dir: str | Path, logger=None) -> None:
        self.config = config
        self.data_cfg = config.data
        self.logger = logger or get_logger(__name__)

        self.dataset_dir = Path(dataset_dir)
        self.pkl_path = self.dataset_dir / config.dataset.pickle_filename
        self.label_mapping = LabelMapping.build(include_none=self.data_cfg.include_none)

        self._raw: RawWaferData | None = None
        self._resized: np.ndarray | None = None
        self._train_indices: np.ndarray | None = None
        self.train_dataset: WaferMapDataset | None = None
        self.val_dataset: WaferMapDataset | None = None
        self.test_dataset: WaferMapDataset | None = None

    # -- Lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        """Load, clean, resize (once) and split the dataset into three views."""
        raw = load_wm811k(
            self.pkl_path,
            self.label_mapping,
            include_none=self.data_cfg.include_none,
            max_samples=self.data_cfg.max_samples,
            seed=self.config.seed,
        )
        self._raw = raw

        self.logger.info("Resizing %d wafer maps to %dpx (one-off)...", len(raw), self.data_cfg.image_size)
        self._resized = resize_wafer_stack(raw.wafer_maps, self.data_cfg.image_size, self.logger)

        train_idx, val_idx, test_idx = self._split(raw)
        self._train_indices = train_idx
        self._log_split(train_idx, val_idx, test_idx)

        common = dict(
            resized_maps=self._resized,
            labels=raw.labels,
            representation=self.data_cfg.representation,
        )
        augment = self.data_cfg.augment
        self.train_dataset = WaferMapDataset(indices=train_idx, augment=augment, **common)
        self.val_dataset = WaferMapDataset(indices=val_idx, augment=False, **common)
        self.test_dataset = WaferMapDataset(indices=test_idx, augment=False, **common)

    # -- Properties ---------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return self.label_mapping.num_classes

    @property
    def input_channels(self) -> int:
        return num_input_channels(self.data_cfg.representation)

    @property
    def wafer_maps(self) -> list[np.ndarray]:
        """The raw (unresized) wafer maps, for e.g. inference demos."""
        if self._raw is None:
            raise RuntimeError("Call setup() before accessing wafer_maps.")
        return self._raw.wafer_maps

    # -- Dataloaders --------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.train_dataset), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.val_dataset), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._require(self.test_dataset), shuffle=False)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from the *training* split (mean 1.0)."""
        if self._raw is None or self._train_indices is None:
            raise RuntimeError("Call setup() before requesting class weights.")

        counts = np.bincount(
            self._raw.labels[self._train_indices], minlength=self.num_classes
        ).astype(np.float64)

        weights = np.ones(self.num_classes, dtype=np.float64)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    # -- Splitting ----------------------------------------------------------
    def _split(self, raw: RawWaferData) -> Split:
        strategy = (self.data_cfg.split_strategy or "lot").lower()
        if strategy == "random":
            return self._split_random(raw.labels)
        if strategy == "lot":
            return self._split_grouped(raw.labels, raw.lot_names)
        if strategy == "official":
            return self._split_official(raw)
        raise ValueError(
            f"Unknown split_strategy {strategy!r}; expected lot | official | random."
        )

    def _split_random(self, labels: np.ndarray) -> Split:
        pos = np.arange(len(labels))
        trainval, test = self._stratified_holdout(pos, labels, self.data_cfg.test_split)
        val_rel = self._val_relative()
        train, val = self._stratified_holdout(trainval, labels[trainval], val_rel)
        return train, val, test

    def _split_grouped(self, labels: np.ndarray, groups: np.ndarray) -> Split:
        pos = np.arange(len(labels))
        trainval, test = self._group_holdout(pos, groups, self.data_cfg.test_split)
        val_rel = self._val_relative()
        train, val = self._group_holdout(trainval, groups, val_rel)
        self._assert_no_group_overlap(groups, train, val, test)
        return train, val, test

    def _split_official(self, raw: RawWaferData) -> Split:
        pos = np.arange(len(raw))
        split = raw.official_split
        test = pos[split == "Test"]
        pool = pos[split != "Test"]  # Training + any rows lacking a flag

        if len(test) == 0:
            self.logger.warning(
                "split_strategy='official' but no 'Test' rows found; "
                "falling back to a lot-aware split."
            )
            return self._split_grouped(raw.labels, raw.lot_names)

        # Carve validation out of the training pool, still lot-aware. Note: the
        # dataset's official Test flag is not guaranteed to be lot-disjoint from
        # Training, so we only enforce (and can only guarantee) train vs val
        # lot-disjointness here. For strict leakage-free evaluation use
        # split_strategy='lot'.
        val_target = self.data_cfg.val_split * len(raw)
        val_rel = min(0.9, val_target / len(pool)) if len(pool) else 0.0
        train, val = self._group_holdout(pool, raw.lot_names, val_rel)
        self._assert_no_group_overlap(raw.lot_names, train, val)
        return train, val, test

    def _val_relative(self) -> float:
        remaining = 1.0 - self.data_cfg.test_split
        return self.data_cfg.val_split / remaining if remaining > 0 else 0.0

    def _stratified_holdout(
        self, pos: np.ndarray, labels: np.ndarray, fraction: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if fraction <= 0:
            return pos, np.array([], dtype=pos.dtype)
        try:
            left, right = train_test_split(
                pos, test_size=fraction, random_state=self.config.seed, stratify=labels
            )
        except ValueError:
            self.logger.warning("Stratified split infeasible (rare classes); using random.")
            left, right = train_test_split(
                pos, test_size=fraction, random_state=self.config.seed
            )
        return left, right

    def _group_holdout(
        self, pos: np.ndarray, groups: np.ndarray, fraction: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if fraction <= 0 or len(pos) == 0:
            return pos, np.array([], dtype=pos.dtype)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=fraction, random_state=self.config.seed
        )
        left_rel, right_rel = next(splitter.split(pos, groups=groups[pos]))
        return pos[left_rel], pos[right_rel]

    def _assert_no_group_overlap(self, groups: np.ndarray, *splits: np.ndarray) -> None:
        """Defensive check that no lot appears in more than one split."""
        seen: set = set()
        for split in splits:
            lots = set(groups[split].tolist())
            overlap = seen & lots
            if overlap:  # pragma: no cover - guards against a splitter regression
                raise RuntimeError(f"Lot leakage across splits: {sorted(overlap)[:5]}")
            seen |= lots

    # -- Internals ----------------------------------------------------------
    def _make_loader(self, dataset: WaferMapDataset, shuffle: bool) -> DataLoader:
        pin_memory = self.data_cfg.pin_memory and torch.cuda.is_available()
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.data_cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _log_split(self, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
        self.logger.info(
            "Split (%s) -> train=%d, val=%d, test=%d",
            self.data_cfg.split_strategy,
            len(train),
            len(val),
            len(test),
        )

    @staticmethod
    def _require(dataset: WaferMapDataset | None) -> WaferMapDataset:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return dataset
