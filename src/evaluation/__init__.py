"""Evaluation package: metrics, evaluator and plots."""

from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_frame,
)
from src.evaluation.plots import (
    save_confusion_matrix_plot,
    save_misclassified_grid,
    save_per_class_f1,
    save_reliability_diagram,
    save_training_curves,
)

__all__ = [
    "Evaluator",
    "compute_metrics",
    "classification_report_dict",
    "confusion_matrix_frame",
    "save_confusion_matrix_plot",
    "save_training_curves",
    "save_per_class_f1",
    "save_misclassified_grid",
    "save_reliability_diagram",
]
