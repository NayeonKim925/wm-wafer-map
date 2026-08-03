"""Inference: rebuild a trained model from a checkpoint and classify wafer maps."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import ModelConfig
from src.data.labels import LabelMapping
from src.data.preprocessing import num_input_channels, preprocess_wafer_map
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_IMAGE_SIZE = 64
_DEFAULT_REPRESENTATION = "onehot"


class Predictor:
    """Classify wafer maps with a trained model loaded from a checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        label_mapping: LabelMapping,
        device: torch.device,
        image_size: int,
        representation: str,
    ) -> None:
        self.model = model.to(device).eval()
        self.label_mapping = label_mapping
        self.device = device
        self.image_size = image_size
        self.representation = representation

    # -- Construction -------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str | torch.device | None = None
    ) -> Predictor:
        """Rebuild an identical model + label mapping from a checkpoint file."""
        resolved_device = (
            resolve_device(device) if isinstance(device, str) or device is None else device
        )
        payload = load_checkpoint(checkpoint_path, map_location=resolved_device)

        label_mapping = LabelMapping.from_list(payload["label_classes"])
        data_cfg = payload.get("config", {}).get("data", {})
        image_size = int(data_cfg.get("image_size", _DEFAULT_IMAGE_SIZE))
        representation = str(data_cfg.get("representation", _DEFAULT_REPRESENTATION))

        model_config = ModelConfig(**payload["model_config"])
        model = build_model(
            model_config,
            in_channels=num_input_channels(representation),
            num_classes=label_mapping.num_classes,
        )
        model.load_state_dict(payload["model_state"])
        logger.info(
            "Loaded predictor from %s (epoch %s).", checkpoint_path, payload.get("epoch", "?")
        )
        return cls(model, label_mapping, resolved_device, image_size, representation)

    # -- Prediction ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, wafer_maps: np.ndarray | Sequence[np.ndarray], top_k: int = 3) -> list[dict]:
        """Classify one or more wafer maps.

        Accepts a single 2-D array or a sequence of 2-D arrays.  Returns one
        result dict per input with the predicted class, confidence, and the
        ``top_k`` class probabilities.
        """
        maps = self._as_list(wafer_maps)
        if not maps:
            return []

        batch = torch.stack(
            [
                torch.from_numpy(
                    np.ascontiguousarray(
                        preprocess_wafer_map(w, self.image_size, self.representation)
                    )
                )
                for w in maps
            ]
        ).to(self.device)

        probs = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        return [self._format(prob, top_k) for prob in probs]

    def predict_one(self, wafer_map: np.ndarray, top_k: int = 3) -> dict:
        """Classify a single wafer map."""
        return self.predict([np.asarray(wafer_map)], top_k=top_k)[0]

    # -- Deployment ---------------------------------------------------------
    def export_torchscript(self, path: str | Path) -> Path:
        """Serialise the model to a self-contained TorchScript module.

        Prefers scripting (control-flow-safe); falls back to tracing on a dummy
        input. The result loads with ``torch.jit.load`` and needs none of this
        codebase — suitable for a serving runtime.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        channels = num_input_channels(self.representation)
        example = torch.zeros(1, channels, self.image_size, self.image_size, device=self.device)
        self.model.eval()
        try:
            module = torch.jit.script(self.model)
        except Exception as exc:  # noqa: BLE001 - tracing is a valid fallback
            logger.warning("torch.jit.script failed (%s); falling back to tracing.", exc)
            module = torch.jit.trace(self.model, example)
        module.save(str(out))
        logger.info("Exported TorchScript model to %s", out)
        return out

    # -- Helpers ------------------------------------------------------------
    @staticmethod
    def _as_list(wafer_maps: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
        if isinstance(wafer_maps, np.ndarray) and wafer_maps.ndim == 2:
            return [wafer_maps]
        return [np.asarray(w) for w in wafer_maps]

    def _format(self, prob: np.ndarray, top_k: int) -> dict:
        k = max(1, min(top_k, self.label_mapping.num_classes))
        order = np.argsort(prob)[::-1]
        top = order[:k]
        best = int(order[0])
        return {
            "predicted_index": best,
            "predicted_class": self.label_mapping.name(best),
            "confidence": float(prob[best]),
            "top_k": [
                {"class": self.label_mapping.name(int(i)), "probability": float(prob[i])}
                for i in top
            ],
        }


def load_wafer_maps(path: str | Path) -> list[np.ndarray]:
    """Load wafer map(s) from a ``.npy`` or ``.pkl`` file for batch inference.

    Supports:
      * ``.npy`` holding a single 2-D array or an object array of 2-D arrays,
      * ``.pkl`` holding a DataFrame with a ``waferMap`` column, or a list/array
        of wafer maps.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        loaded = np.load(p, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 2:
            return [loaded]
        return [np.asarray(w) for w in loaded]

    if suffix in (".pkl", ".pickle"):
        import pandas as pd

        obj = pd.read_pickle(p)
        if isinstance(obj, pd.DataFrame):
            if "waferMap" not in obj.columns:
                raise KeyError("Pickle DataFrame has no 'waferMap' column.")
            return [np.asarray(w) for w in obj["waferMap"].tolist()]
        return [np.asarray(w) for w in obj]

    raise ValueError(f"Unsupported input format {suffix!r}; use .npy or .pkl.")
