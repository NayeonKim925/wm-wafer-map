"""Grad-CAM visualisation: overlay attention heatmaps on wafer maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data.preprocessing import preprocess_wafer_map, resize_wafer_map
from src.interpretability.gradcam import GradCAM
from src.utils.logging import get_logger
from src.utils.plotting import get_pyplot, prepare_output

if TYPE_CHECKING:  # avoid a runtime import cycle
    from src.inference import Predictor

logger = get_logger(__name__)


def plot_gradcam_gallery(
    predictor: Predictor,
    wafer_maps: list[np.ndarray],
    path: str | Path,
    true_labels: np.ndarray | None = None,
    max_samples: int = 8,
    columns: int = 4,
    seed: int = 42,
) -> Path | None:
    """Grid of wafer maps with Grad-CAM heatmaps overlaid on the predicted class.

    Green/red titles mark correct/incorrect predictions when ``true_labels`` is
    provided. Returns the output path, or ``None`` if matplotlib is unavailable.
    """
    plt = get_pyplot()
    if plt is None or not wafer_maps:
        return None
    out = prepare_output(path)

    rng = np.random.default_rng(seed)
    n = min(max_samples, len(wafer_maps))
    picks = rng.choice(len(wafer_maps), size=n, replace=False)

    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.2 * rows), squeeze=False)
    size = predictor.image_size

    with GradCAM(predictor.model) as cam:
        for cell, idx in enumerate(picks):
            wafer = wafer_maps[int(idx)]
            tensor = torch.from_numpy(
                np.ascontiguousarray(preprocess_wafer_map(wafer, size, predictor.representation))
            ).unsqueeze(0).to(predictor.device)

            heatmap = cam.overlay(tensor, size)[0]
            result = predictor.predict_one(wafer)

            ax = axes[cell // columns][cell % columns]
            background = resize_wafer_map(wafer, size)
            ax.imshow(background, cmap="gray_r", vmin=0, vmax=2, interpolation="nearest")
            ax.imshow(heatmap, cmap="jet", alpha=0.5, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])

            title = f"{result['predicted_class']} ({result['confidence']:.2f})"
            color = "black"
            if true_labels is not None:
                truth = predictor.label_mapping.name(int(true_labels[int(idx)]))
                correct = truth == result["predicted_class"]
                color = "#2E7D32" if correct else "#C62828"
                title = f"{'✓' if correct else '✗'} {title}\ntrue: {truth}"
            ax.set_title(title, fontsize=9, color=color)

    for cell in range(n, rows * columns):
        axes[cell // columns][cell % columns].axis("off")

    fig.suptitle("Grad-CAM: where the model looks", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Grad-CAM gallery to %s", out)
    return out
