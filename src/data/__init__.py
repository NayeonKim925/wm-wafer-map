"""Data acquisition and pipeline package."""

from src.data.datamodule import WaferDataModule
from src.data.dataset import WaferMapDataset, load_wm811k
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
)

__all__ = [
    "WaferDataModule",
    "WaferMapDataset",
    "load_wm811k",
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
    "encode_wafer_map",
    "normalize_failure_type",
    "num_input_channels",
]
