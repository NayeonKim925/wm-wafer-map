#!/usr/bin/env python3
"""Exploratory data analysis for WM-811K.

Downloads the dataset if needed, then writes a class-distribution chart and a
grid of sample wafer maps per class.

    python scripts/visualize_dataset.py
    python scripts/visualize_dataset.py --examples 8 --output-dir outputs/eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.data import DatasetError, DatasetManager, LabelMapping, load_wm811k  # noqa: E402
from src.data.visualize import plot_class_distribution, plot_sample_wafers  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the WM-811K dataset (class balance + sample wafers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--examples", type=int, default=5, help="Sample wafers per class.")
    parser.add_argument("--output-dir", default=None, help="Where to write figures.")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = Config.load(args.config, overrides=args.overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(args.log_level or config.log_level)
    logger = get_logger("visualize")

    manager = DatasetManager(config.dataset, logger=logger)
    try:
        dataset_dir = manager.prepare()
    except DatasetError as exc:
        logger.error("Could not obtain the dataset.\n%s", exc)
        return 1

    mapping = LabelMapping.build(include_none=config.data.include_none)
    raw = load_wm811k(
        dataset_dir / config.dataset.pickle_filename,
        mapping,
        include_none=config.data.include_none,
        max_samples=config.data.max_samples,
        seed=config.seed,
    )

    output_dir = resolve_path(args.output_dir) if args.output_dir else (
        resolve_path(config.output.root_dir) / "eda"
    )
    dist = plot_class_distribution(raw.labels, mapping, output_dir / "class_distribution.png")
    grid = plot_sample_wafers(
        raw.wafer_maps, raw.labels, mapping, output_dir / "sample_wafers.png",
        examples_per_class=args.examples, seed=config.seed,
    )

    print(f"\nEDA complete. Figures written to {output_dir}")
    for artefact in (dist, grid):
        if artefact is not None:
            print(f"  {artefact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
