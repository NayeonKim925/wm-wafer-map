"""Exploratory visualisations of the WM-811K dataset.

These help a reader understand the data before trusting any model: how
imbalanced the classes are, and what each defect pattern actually looks like.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.labels import LabelMapping
from src.utils.logging import get_logger
from src.utils.plotting import get_pyplot, prepare_output

logger = get_logger(__name__)

# Wafer cells: 0 = background, 1 = passing die, 2 = failing die.
_WAFER_COLORS = ["#f0f0f0", "#4C72B0", "#C44E52"]


def plot_class_distribution(
    labels: np.ndarray, label_mapping: LabelMapping, path: str | Path
) -> Path | None:
    """Horizontal bar chart of class counts (log-friendly for imbalance)."""
    plt = get_pyplot()
    if plt is None:
        return None
    out = prepare_output(path)

    counts = np.bincount(labels, minlength=label_mapping.num_classes)
    order = np.argsort(counts)
    names = [label_mapping.name(int(i)) for i in order]
    values = counts[order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * label_mapping.num_classes + 1.5))
    ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("count")
    ax.set_title(f"Class distribution (n={int(counts.sum())})")
    for i, value in enumerate(values):
        ax.text(value, i, f" {int(value)}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved class-distribution chart to %s", out)
    return out


def plot_sample_wafers(
    wafer_maps: list[np.ndarray],
    labels: np.ndarray,
    label_mapping: LabelMapping,
    path: str | Path,
    examples_per_class: int = 5,
    seed: int = 42,
) -> Path | None:
    """Grid of example wafer maps, one row per class."""
    plt = get_pyplot()
    if plt is None:
        return None
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out = prepare_output(path)
    rng = np.random.default_rng(seed)
    cmap = ListedColormap(_WAFER_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    num_classes = label_mapping.num_classes
    fig, axes = plt.subplots(
        num_classes,
        examples_per_class,
        figsize=(1.6 * examples_per_class, 1.6 * num_classes),
        squeeze=False,
    )

    for row in range(num_classes):
        candidates = np.where(labels == row)[0]
        chosen = (
            rng.choice(candidates, size=min(examples_per_class, len(candidates)), replace=False)
            if len(candidates)
            else []
        )
        for col in range(examples_per_class):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(chosen):
                ax.imshow(np.asarray(wafer_maps[int(chosen[col])]), cmap=cmap, norm=norm)
            else:
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(label_mapping.name(row), rotation=0, ha="right", va="center", fontsize=9)

    fig.suptitle("WM-811K sample wafer maps by class", y=1.002)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample-wafer grid to %s", out)
    return out
