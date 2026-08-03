#!/usr/bin/env python3
"""Standalone WM-811K dataset downloader.

Downloads the dataset from Kaggle (if not already present), verifies its
integrity, and prints its on-disk location.  Safe to re-run: an existing,
verified copy is left untouched.

Examples
--------
    python scripts/download_dataset.py
    python scripts/download_dataset.py --config configs/default.yaml
    python scripts/download_dataset.py --force
    python scripts/download_dataset.py --set dataset.link_mode=copy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.data import DatasetError, DatasetManager  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the WM-811K wafer-map dataset.",
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
        help="Override config values, e.g. --set dataset.link_mode=copy",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the configured log level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    overrides = list(args.overrides)
    if args.force:
        overrides.append("dataset.force_download=true")

    try:
        config = Config.load(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(args.log_level or config.log_level)
    logger = get_logger("download")

    manager = DatasetManager(config.dataset, logger=logger)

    try:
        dataset_dir = manager.prepare()
    except DatasetError as exc:
        logger.error("Dataset preparation failed.\n%s", exc)
        return 1

    # User-facing CLI summary (intentional stdout output).
    info = manager.describe()
    print("\n" + "=" * 70)
    print("WM-811K dataset is ready.")
    print(f"  Kaggle handle : {info['kaggle_handle']}")
    print(f"  Location      : {dataset_dir}")
    for entry in info["files"]:
        status = "ok" if entry["exists"] else "MISSING"
        print(f"  File          : {entry['name']}  [{status}, {entry['size_mb']} MB]")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
