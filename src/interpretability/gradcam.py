"""Grad-CAM explainability for the wafer-map classifiers.

Grad-CAM [Selvaraju et al., 2017] highlights the spatial regions a CNN relies on
for a prediction. For wafer maps this is a sanity check that the model attends to
the actual defect signature (a central blob, an edge ring, a scratch line) rather
than a spurious artefact.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_target_layer(model: nn.Module) -> nn.Conv2d:
    """Return the last ``Conv2d`` in ``model`` (the default Grad-CAM target)."""
    target: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target = module
    if target is None:
        raise ValueError("No Conv2d layer found; Grad-CAM needs a convolutional model.")
    return target


class GradCAM:
    """Grad-CAM for a CNN, hooking a single convolutional target layer.

    Use as a context manager (or call :meth:`remove` when done) so the forward /
    backward hooks are cleaned up.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activations),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self, inputs: torch.Tensor, class_indices: torch.Tensor | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cams, class_indices)`` for a batch.

        ``cams`` has shape ``(B, h, w)`` at the target layer's resolution, each map
        min-max normalised to ``[0, 1]``. If ``class_indices`` is ``None`` the
        predicted class is explained.
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(inputs)
        if class_indices is None:
            class_indices = logits.argmax(dim=1)
        score = logits.gather(1, class_indices.view(-1, 1)).sum()
        score.backward()

        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1))  # (B, h, w)

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

        if was_training:
            self.model.train()
        return cam.cpu().numpy(), class_indices.detach().cpu().numpy()

    def overlay(self, inputs: torch.Tensor, size: int) -> np.ndarray:
        """Return CAMs upsampled (bilinear) to ``size x size`` for display."""
        cams, _ = self(inputs)
        tensor = torch.from_numpy(cams).unsqueeze(1)
        upsampled = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return upsampled.squeeze(1).numpy()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
