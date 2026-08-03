"""Shared command-line helpers for the entry-point scripts.

Keeps argument parsing, config loading and experiment-directory creation in one
place so ``train.py`` / ``evaluate.py`` do not duplicate boilerplate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.utils.paths import resolve_path


def common_parser(description: str) -> argparse.ArgumentParser:
    """An argument parser preloaded with the config-related flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.epochs=5 data.batch_size=64",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the configured log level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def load_config_from_args(
    args: argparse.Namespace, extra_overrides: list[str] | None = None
) -> Config:
    """Load a :class:`Config` from parsed CLI args plus any programmatic overrides."""
    overrides = list(getattr(args, "overrides", []) or [])
    overrides.extend(extra_overrides or [])
    return Config.load(getattr(args, "config", None), overrides=overrides)


def create_experiment_dir(config: Config) -> Path:
    """Create (and return) a unique output directory for this run."""
    base = resolve_path(config.output.root_dir) / config.output.experiment_name
    if config.output.timestamp_subdir:
        base = base / datetime.now().strftime("%Y%m%d-%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return base
