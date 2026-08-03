"""Model interpretability: Grad-CAM attention maps."""

from src.interpretability.gradcam import GradCAM, find_target_layer
from src.interpretability.visualize import plot_gradcam_gallery

__all__ = ["GradCAM", "find_target_layer", "plot_gradcam_gallery"]
