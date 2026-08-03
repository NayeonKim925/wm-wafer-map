"""Cross-cutting utilities: paths, logging, seeding, device, checkpoints."""

from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import resolve_device
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import get_project_root, resolve_path
from src.utils.seed import set_seed

__all__ = [
    "get_logger",
    "setup_logging",
    "get_project_root",
    "resolve_path",
    "resolve_device",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
]
