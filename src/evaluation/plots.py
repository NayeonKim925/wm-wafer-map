"""Optional visualisation helpers.

matplotlib is imported lazily so the core pipeline does not hard-depend on it;
if it is unavailable, plotting is skipped with a warning rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import LabelMapping
from src.utils.logging import get_logger
from src.utils.plotting import get_pyplot, prepare_output

logger = get_logger(__name__)

# Wafer cells: 0 = background, 1 = passing die, 2 = failing die.
_WAFER_COLORS = ["#f0f0f0", "#4C72B0", "#C44E52"]


def save_confusion_matrix_plot(
    cm: pd.DataFrame, path: str | Path, normalize: bool = True
) -> Path | None:
    """Render a confusion matrix heatmap to ``path``; return the path or ``None``."""
    plt = get_pyplot()
    if plt is None:
        return None
    out = prepare_output(path)

    matrix = cm.to_numpy().astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    classes = list(cm.index)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 2))
    image = ax.imshow(
        matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix" if normalize else "Confusion matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion-matrix plot to %s", out)
    return out


def save_training_curves(history: list[dict], path: str | Path) -> Path | None:
    """Plot train/val loss and macro-F1 across epochs from the run history."""
    plt = get_pyplot()
    if plt is None or not history:
        return None
    out = prepare_output(path)

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [h.get("loss") for h in history], "-o", label="train", ms=3)
    ax_loss.plot(epochs, [h.get("val_loss") for h in history], "-o", label="val", ms=3)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_f1.plot(epochs, [h.get("f1_macro") for h in history], "-o", label="train", ms=3)
    ax_f1.plot(epochs, [h.get("val_f1_macro") for h in history], "-o", label="val", ms=3)
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro-F1")
    ax_f1.set_title("Macro-F1")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to %s", out)
    return out


def save_per_class_f1(report: dict, classes: list[str], path: str | Path) -> Path | None:
    """Bar chart of per-class F1 from a sklearn classification report dict."""
    plt = get_pyplot()
    if plt is None:
        return None
    out = prepare_output(path)

    scores = [float(report.get(name, {}).get("f1-score", 0.0)) for name in classes]
    order = np.argsort(scores)
    sorted_classes = [classes[i] for i in order]
    sorted_scores = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(classes) + 1.5))
    bars = ax.barh(sorted_classes, sorted_scores, color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1")
    for bar, score in zip(bars, sorted_scores):
        ax.text(min(score + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved per-class F1 chart to %s", out)
    return out


def save_misclassified_grid(
    wafer_maps: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: LabelMapping,
    path: str | Path,
    max_examples: int = 12,
    columns: int = 4,
    seed: int = 42,
) -> Path | None:
    """Montage of misclassified wafers (title: true -> predicted). Error analysis."""
    plt = get_pyplot()
    if plt is None:
        return None
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from src.data.preprocessing import resize_wafer_map

    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0]
    if len(wrong) == 0:
        logger.info("No misclassifications to plot.")
        return None

    out = prepare_output(path)
    rng = np.random.default_rng(seed)
    picks = rng.choice(wrong, size=min(max_examples, len(wrong)), replace=False)

    cmap = ListedColormap(_WAFER_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    rows = int(np.ceil(len(picks) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.8 * rows), squeeze=False)

    for cell in range(rows * columns):
        ax = axes[cell // columns][cell % columns]
        ax.set_xticks([])
        ax.set_yticks([])
        if cell < len(picks):
            idx = int(picks[cell])
            display = resize_wafer_map(np.asarray(wafer_maps[idx]), 64)
            ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(
                f"true: {label_mapping.name(int(y_true[idx]))}\n"
                f"pred: {label_mapping.name(int(y_pred[idx]))}",
                fontsize=8,
                color="#C62828",
            )
        else:
            ax.axis("off")

    fig.suptitle("Error analysis: misclassified wafers", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved misclassified grid to %s", out)
    return out


def save_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray, path: str | Path, n_bins: int = 10
) -> Path | None:
    """Calibration (reliability) diagram with the Expected Calibration Error.

    Shows whether predicted confidence matches empirical accuracy — a signal of
    model calibration that a plain accuracy number hides.
    """
    plt = get_pyplot()
    if plt is None:
        return None
    out = prepare_output(path)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidence = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accuracies, ece = [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        centres.append((lo + hi) / 2)
        if mask.any():
            acc = correct[mask].mean()
            accuracies.append(acc)
            ece += (mask.mean()) * abs(acc - confidence[mask].mean())
        else:
            accuracies.append(np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.bar(centres, accuracies, width=1.0 / n_bins * 0.9, alpha=0.75,
           color="#4C72B0", edgecolor="black", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved reliability diagram to %s (ECE=%.3f)", out, ece)
    return out
