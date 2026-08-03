"""Data acquisition and pipeline package."""

from src.data.datamodule import WaferDataModule
from src.data.dataset import (
    RawWaferData,
    WaferMapDataset,
    augment_wafer_map,
    load_wm811k,
    seed_worker,
)
from src.data.dataset_manager import (
    DatasetDownloadError,
    DatasetError,
    DatasetManager,
    DatasetVerificationError,
)
from src.data.labels import (
    ALL_CLASSES,
    DEFECT_CLASSES,
    NONE_CLASS,
    LabelMapping,
    canonicalize_label,
)
from src.data.preprocessing import (
    encode_wafer_map,
    normalize_failure_type,
    num_input_channels,
    preprocess_wafer_map,
    resize_wafer_map,
    resize_wafer_stack,
)

__all__ = [
    "WaferDataModule",
    "WaferMapDataset",
    "RawWaferData",
    "load_wm811k",
    "augment_wafer_map",
    "seed_worker",
    "DatasetManager",
    "DatasetError",
    "DatasetDownloadError",
    "DatasetVerificationError",
    "LabelMapping",
    "canonicalize_label",
    "ALL_CLASSES",
    "DEFECT_CLASSES",
    "NONE_CLASS",
    "preprocess_wafer_map",
    "resize_wafer_map",
    "resize_wafer_stack",
    "encode_wafer_map",
    "normalize_failure_type",
    "num_input_channels",
]
