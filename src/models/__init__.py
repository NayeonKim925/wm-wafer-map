"""Model architectures and the build registry."""

from src.models.cnn import WaferCNN
from src.models.factory import available_models, build_model, register_model
from src.models.resnet import WaferResNet

__all__ = [
    "WaferCNN",
    "WaferResNet",
    "build_model",
    "register_model",
    "available_models",
]
