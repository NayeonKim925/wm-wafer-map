"""Tests for label handling and wafer-map preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import augment_wafer_map, seed_worker
from src.data.labels import LabelMapping, canonicalize_label
from src.data.preprocessing import (
    encode_wafer_map,
    normalize_failure_type,
    num_input_channels,
    preprocess_wafer_map,
    resize_wafer_map,
    resize_wafer_stack,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.array([["Center"]], dtype=object), "Center"),
        (np.array([["none"]], dtype=object), "none"),
        (np.array([[]], dtype=object), None),
        ([["Scratch"]], "Scratch"),
        ("Donut", "Donut"),
        (float("nan"), None),
        (None, None),
    ],
)
def test_normalize_failure_type(value, expected):
    assert normalize_failure_type(value) == expected


def test_canonicalize_label_case_insensitive():
    assert canonicalize_label("edge-ring") == "Edge-Ring"
    assert canonicalize_label("  NONE ") == "none"
    assert canonicalize_label("not-a-class") is None
    assert canonicalize_label(np.nan) is None


def test_resize_preserves_categorical_values():
    wafer = np.array([[0, 1], [2, 1]], dtype=np.int8)
    resized = resize_wafer_map(wafer, 8)
    assert resized.shape == (8, 8)
    assert set(np.unique(resized)).issubset({0, 1, 2})


def test_resize_rejects_non_2d():
    with pytest.raises(ValueError):
        resize_wafer_map(np.zeros((3, 3, 3)), 8)


def test_encode_scalar_and_onehot():
    resized = np.array([[0, 1], [2, 0]], dtype=np.int8)

    scalar = encode_wafer_map(resized, "scalar")
    assert scalar.shape == (1, 2, 2)
    assert scalar.dtype == np.float32
    assert scalar.max() <= 1.0 and scalar.min() >= 0.0

    onehot = encode_wafer_map(resized, "onehot")
    assert onehot.shape == (3, 2, 2)
    # exactly one channel active per cell
    assert np.all(onehot.sum(axis=0) == 1)


def test_num_input_channels():
    assert num_input_channels("scalar") == 1
    assert num_input_channels("onehot") == 3
    with pytest.raises(ValueError):
        num_input_channels("bogus")


def test_preprocess_pipeline_shape():
    wafer = np.random.default_rng(0).integers(0, 3, size=(31, 27)).astype(np.int8)
    out = preprocess_wafer_map(wafer, image_size=32, representation="onehot")
    assert out.shape == (3, 32, 32)


def test_resize_wafer_stack_is_compact_uint8():
    rng = np.random.default_rng(0)
    maps = [rng.integers(0, 3, size=(rng.integers(20, 40), rng.integers(20, 40))) for _ in range(5)]
    stack = resize_wafer_stack(maps, image_size=16)
    assert stack.shape == (5, 16, 16)
    assert stack.dtype == np.uint8
    assert set(np.unique(stack)).issubset({0, 1, 2})


def test_augment_preserves_shape_and_values():
    wafer = np.random.default_rng(1).integers(0, 3, size=(16, 16)).astype(np.uint8)
    out = augment_wafer_map(wafer)
    assert out.shape == wafer.shape
    assert set(np.unique(out)).issubset({0, 1, 2})


def test_augment_produces_variety():
    """Over many draws the augmentation should not be a constant identity."""
    wafer = np.arange(16, dtype=np.uint8).reshape(4, 4)
    np.random.seed(0)
    outs = {augment_wafer_map(wafer).tobytes() for _ in range(50)}
    assert len(outs) > 1


def test_seed_worker_makes_numpy_deterministic():
    import torch

    torch.manual_seed(123)
    seed_worker(0)
    a = np.random.rand(5)
    torch.manual_seed(123)
    seed_worker(0)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_label_mapping_roundtrip():
    mapping = LabelMapping.build(include_none=True)
    assert mapping.num_classes == 9
    assert mapping.name(mapping.index("Scratch")) == "Scratch"

    without_none = LabelMapping.build(include_none=False)
    assert without_none.num_classes == 8
    assert "none" not in without_none
