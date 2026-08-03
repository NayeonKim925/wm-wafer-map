"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np

from src.data.labels import LabelMapping
from src.evaluation.metrics import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_frame,
)


def test_compute_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 2, 2, 1, 0, 3, 3])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 3, 2])
    metrics = compute_metrics(y_true, y_pred)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "cohen_kappa",
        "precision_macro",
        "recall_macro",
    ):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 3])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0


def test_confusion_matrix_shape_and_report():
    mapping = LabelMapping.build(include_none=True)
    y_true = np.arange(mapping.num_classes)
    y_pred = np.arange(mapping.num_classes)
    cm = confusion_matrix_frame(y_true, y_pred, mapping)
    assert cm.shape == (mapping.num_classes, mapping.num_classes)
    assert list(cm.index) == mapping.classes

    report = classification_report_dict(y_true, y_pred, mapping)
    assert "Center" in report
    assert report["Center"]["f1-score"] == 1.0
