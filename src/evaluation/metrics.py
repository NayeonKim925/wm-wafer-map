"""Classification metrics shared by training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import classification_report as sk_classification_report

from src.data.labels import LabelMapping


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return headline scalar metrics (accuracy + macro/weighted P/R/F1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, label_mapping: LabelMapping
) -> dict:
    """Per-class precision/recall/F1 as a nested dict."""
    labels = list(range(label_mapping.num_classes))
    return sk_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_mapping.classes,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_frame(
    y_true: np.ndarray, y_pred: np.ndarray, label_mapping: LabelMapping
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    labels = list(range(label_mapping.num_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=label_mapping.classes, columns=label_mapping.classes)
