"""Optional visualisation helpers.

matplotlib is imported lazily so the core pipeline does not hard-depend on it;
if it is unavailable, plotting is skipped with a warning rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_confusion_matrix_plot(cm: pd.DataFrame, path: str | Path, normalize: bool = True) -> Path | None:
    """Render a confusion matrix heatmap to ``path``; return the path or ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping confusion-matrix plot.")
        return None

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    matrix = cm.to_numpy().astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    classes = list(cm.index)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 2))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
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
