"""Inference visualisation: annotated prediction galleries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.data.preprocessing import resize_wafer_map
from src.utils.logging import get_logger
from src.utils.plotting import get_pyplot, prepare_output

if TYPE_CHECKING:
    from src.inference import Predictor

logger = get_logger(__name__)

# Wafer cells: 0 = background, 1 = passing die, 2 = failing die.
_WAFER_COLORS = ["#f0f0f0", "#4C72B0", "#C44E52"]


def plot_prediction_gallery(
    predictor: Predictor,
    wafer_maps: list[np.ndarray],
    path: str | Path,
    true_labels: np.ndarray | None = None,
    max_samples: int = 8,
    columns: int = 2,
    top_k: int = 3,
    seed: int = 42,
) -> Path | None:
    """Grid of wafer maps, each with its prediction and a top-k probability bar.

    When ``true_labels`` is given, titles are green/red for correct/incorrect.
    """
    plt = get_pyplot()
    if plt is None or not wafer_maps:
        return None
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out = prepare_output(path)
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(wafer_maps))
    picks = rng.choice(len(wafer_maps), size=n, replace=False)
    results = predictor.predict([wafer_maps[int(i)] for i in picks], top_k=top_k)

    cmap = ListedColormap(_WAFER_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns * 2, figsize=(5.0 * columns, 2.6 * rows), squeeze=False)

    for cell, (idx, result) in enumerate(zip(picks, results)):
        r, c = cell // columns, cell % columns
        ax_img, ax_bar = axes[r][2 * c], axes[r][2 * c + 1]

        display = resize_wafer_map(wafer_maps[int(idx)], predictor.image_size)
        ax_img.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        color = "black"
        title = f"pred: {result['predicted_class']} ({result['confidence']:.2f})"
        if true_labels is not None:
            truth = predictor.label_mapping.name(int(true_labels[int(idx)]))
            correct = truth == result["predicted_class"]
            color = "#2E7D32" if correct else "#C62828"
            title = f"{'✓' if correct else '✗'} " + title + f"\ntrue: {truth}"
        ax_img.set_title(title, fontsize=9, color=color)

        names = [c["class"] for c in result["top_k"]][::-1]
        probs = [c["probability"] for c in result["top_k"]][::-1]
        bar_colors = ["#C62828" if name == result["predicted_class"] else "#B0B7C3" for name in names]
        ax_bar.barh(names, probs, color=bar_colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.tick_params(labelsize=8)
        ax_bar.set_xlabel("probability", fontsize=8)

    for cell in range(n, rows * columns):
        r, c = cell // columns, cell % columns
        axes[r][2 * c].axis("off")
        axes[r][2 * c + 1].axis("off")

    fig.suptitle("Inference: predictions with class probabilities", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction gallery to %s", out)
    return out
