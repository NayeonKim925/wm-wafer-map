"""Model evaluation: metrics, per-class report and confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.labels import LabelMapping
from src.evaluation.metrics import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_frame,
)
from src.evaluation.plots import save_confusion_matrix_plot, save_per_class_f1
from src.utils.logging import get_logger


class Evaluator:
    """Run a trained model over a dataloader and report classification metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        label_mapping: LabelMapping,
        logger=None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.label_mapping = label_mapping
        self.logger = logger or get_logger(__name__)

    @torch.no_grad()
    def collect_predictions(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(y_true, y_pred, y_prob)`` over the whole loader."""
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        y_prob: list[np.ndarray] = []

        for inputs, labels in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            y_prob.append(probs.cpu().numpy())
            y_pred.append(probs.argmax(1).cpu().numpy())
            y_true.append(labels.numpy())

        return (
            np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(y_prob),
        )

    def evaluate(
        self, loader: DataLoader, output_dir: str | Path | None = None
    ) -> dict:
        """Evaluate on ``loader`` and, if ``output_dir`` is given, save artefacts."""
        y_true, y_pred, _ = self.collect_predictions(loader)

        metrics = compute_metrics(y_true, y_pred)
        report = classification_report_dict(y_true, y_pred, self.label_mapping)
        confusion = confusion_matrix_frame(y_true, y_pred, self.label_mapping)

        self.logger.info(
            "Evaluation | acc=%.4f bal_acc=%.4f f1_macro=%.4f f1_weighted=%.4f kappa=%.4f (n=%d)",
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1_macro"],
            metrics["f1_weighted"],
            metrics["cohen_kappa"],
            len(y_true),
        )

        results = {"metrics": metrics, "report": report, "confusion_matrix": confusion}
        if output_dir is not None:
            self._save(results, Path(output_dir))
        return results

    def _save(self, results: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "classification_report.json").write_text(
            json.dumps(
                {"metrics": results["metrics"], "per_class": results["report"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        results["confusion_matrix"].to_csv(output_dir / "confusion_matrix.csv")
        save_confusion_matrix_plot(
            results["confusion_matrix"], output_dir / "confusion_matrix.png"
        )
        save_per_class_f1(
            results["report"], self.label_mapping.classes, output_dir / "per_class_f1.png"
        )
        self.logger.info("Saved evaluation artefacts to %s", output_dir)
