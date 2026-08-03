"""Automatic experiment reporting.

Assembles a Markdown ``report.md`` for a run: configuration summary, headline
and per-class metrics, and every figure the pipeline produced (training curves,
confusion matrix, per-class F1, prediction gallery, Grad-CAM, error analysis and
a calibration diagram).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import Config
from src.evaluation.plots import save_misclassified_grid, save_reliability_diagram
from src.inference.visualize import plot_prediction_gallery
from src.interpretability import plot_gradcam_gallery
from src.utils.logging import get_logger

# Figure key -> caption, in report display order.
_FIGURE_CAPTIONS = {
    "class_distribution": "Class distribution of the labelled subset.",
    "sample_wafers": "Example wafer maps per class.",
    "training_curves": "Training and validation loss / macro-F1 per epoch.",
    "confusion_matrix": "Row-normalised confusion matrix on the test split.",
    "per_class_f1": "Per-class F1 on the test split.",
    "prediction_gallery": "Sample predictions with class probabilities.",
    "gradcam": "Grad-CAM attention for the predicted class.",
    "misclassified": "Error analysis: misclassified wafers (true vs predicted).",
    "reliability": "Calibration (reliability) diagram with ECE.",
}

_HEADLINE_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("f1_macro", "Macro-F1"),
    ("f1_weighted", "Weighted-F1"),
    ("cohen_kappa", "Cohen's κ"),
]


def generate_visual_report(
    run_dir: str | Path,
    config: Config,
    datamodule,
    predictor,
    eval_results: dict,
    extra_figures: dict[str, Path | None] | None = None,
    logger=None,
) -> Path:
    """Create report figures + ``report.md`` for a completed run.

    Returns the path to the written ``report.md``.
    """
    logger = logger or get_logger(__name__)
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"

    y_true = eval_results["y_true"]
    y_pred = eval_results["y_pred"]
    y_prob = eval_results["y_prob"]

    test_indices = datamodule.test_dataset.indices
    test_wafers = [datamodule.wafer_maps[int(i)] for i in test_indices]
    samples = config.output.report_samples

    figures: dict[str, Path | None] = dict(extra_figures or {})
    figures["prediction_gallery"] = plot_prediction_gallery(
        predictor, test_wafers, report_dir / "prediction_gallery.png",
        true_labels=y_true, max_samples=samples, columns=2, seed=config.seed,
    )
    figures["gradcam"] = plot_gradcam_gallery(
        predictor, test_wafers, report_dir / "gradcam.png",
        true_labels=y_true, max_samples=samples, columns=4, seed=config.seed,
    )
    figures["misclassified"] = save_misclassified_grid(
        test_wafers, y_true, y_pred, datamodule.label_mapping,
        report_dir / "misclassified.png", seed=config.seed,
    )
    figures["reliability"] = save_reliability_diagram(
        y_true, y_prob, report_dir / "reliability.png"
    )

    # Existing pipeline figures (may or may not exist).
    figures.setdefault("training_curves", _if_exists(run_dir / "training_curves.png"))
    figures.setdefault("confusion_matrix", _if_exists(run_dir / "evaluation/confusion_matrix.png"))
    figures.setdefault("per_class_f1", _if_exists(run_dir / "evaluation/per_class_f1.png"))

    report_path = _write_markdown(
        run_dir, config, eval_results["metrics"], eval_results["report"],
        datamodule.label_mapping, figures,
    )
    logger.info("Wrote experiment report to %s", report_path)
    return report_path


def _write_markdown(run_dir, config, metrics, per_class, label_mapping, figures) -> Path:
    lines: list[str] = []
    lines.append(f"# Experiment report — {config.output.experiment_name}")
    lines.append("")
    lines.append(f"_Auto-generated {datetime.now():%Y-%m-%d %H:%M:%S}._")
    lines.append("")

    lines.append("## Run configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Model | `{config.model.name}` (base_channels={config.model.base_channels}) |")
    lines.append(f"| Split strategy | `{config.data.split_strategy}` |")
    lines.append(f"| Image size | {config.data.image_size} |")
    lines.append(f"| Representation | `{config.data.representation}` |")
    lines.append(f"| Epochs | {config.training.epochs} |")
    lines.append(f"| Optimizer | `{config.training.optimizer.name}` (lr={config.training.optimizer.lr}) |")
    lines.append(f"| Classes | {label_mapping.num_classes} |")
    lines.append(f"| Seed | {config.seed} |")
    lines.append("")

    lines.append("## Test metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, name in _HEADLINE_METRICS:
        if key in metrics:
            lines.append(f"| {name} | {metrics[key]:.4f} |")
    lines.append("")

    lines.append("## Per-class F1 (test)")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in label_mapping.classes:
        row = per_class.get(name, {})
        lines.append(
            f"| {name} | {row.get('precision', 0):.3f} | {row.get('recall', 0):.3f} "
            f"| {row.get('f1-score', 0):.3f} | {int(row.get('support', 0))} |"
        )
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for key, caption in _FIGURE_CAPTIONS.items():
        figure = figures.get(key)
        if figure is None:
            continue
        rel = Path(figure).resolve().relative_to(Path(run_dir).resolve())
        lines.append(f"**{caption}**")
        lines.append("")
        lines.append(f"![{key}]({rel.as_posix()})")
        lines.append("")

    report_path = Path(run_dir) / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _if_exists(path: Path) -> Path | None:
    return path if path.is_file() else None
