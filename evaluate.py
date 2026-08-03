#!/usr/bin/env python3
"""Evaluate a trained wafer-map classifier on a data split.

    python evaluate.py --checkpoint outputs/wafer_cnn/<run>/checkpoints/best.pt
    python evaluate.py --checkpoint <ckpt> --split val

By default the configuration stored *inside* the checkpoint is reused, so the
data split and label mapping exactly match training.  Pass ``--config`` to
override.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cli import common_parser, load_config_from_args  # noqa: E402
from src.config import Config, ConfigError, ModelConfig  # noqa: E402
from src.data import DatasetError, DatasetManager, LabelMapping, WaferDataModule  # noqa: E402
from src.evaluation import Evaluator  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("Evaluate a trained WM-811K classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write evaluation artefacts (defaults next to the checkpoint).",
    )
    args = parser.parse_args(argv)

    # Read the checkpoint on CPU first so we can recover its stored config.
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2
    payload = load_checkpoint(checkpoint_path, map_location="cpu")

    try:
        if args.config is not None:
            config = load_config_from_args(args)
        elif payload.get("config"):
            config = Config.from_dict(payload["config"], overrides=args.overrides)
        else:
            config = load_config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(args.log_level or config.log_level)
    logger = get_logger("evaluate")
    device = resolve_device(config.training.device)

    # Ensure data is available and build the requested split.
    manager = DatasetManager(config.dataset, logger=logger)
    try:
        dataset_dir = manager.prepare()
    except DatasetError as exc:
        logger.error("Could not obtain the dataset.\n%s", exc)
        return 1

    datamodule = WaferDataModule(config, dataset_dir, logger=logger)
    datamodule.setup()

    # Rebuild the model straight from the checkpoint (self-describing).
    label_mapping = LabelMapping.from_list(payload["label_classes"])
    model = build_model(
        ModelConfig(**payload["model_config"]),
        in_channels=datamodule.input_channels,
        num_classes=label_mapping.num_classes,
    )
    model.load_state_dict(payload["model_state"])

    loaders = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }
    loader = loaders[args.split]()

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = checkpoint_path.resolve().parent.parent / f"evaluation_{args.split}"

    evaluator = Evaluator(model, device, label_mapping, logger=logger)
    results = evaluator.evaluate(loader, output_dir=output_dir)

    metrics = results["metrics"]
    print(f"\nEvaluation on '{args.split}' split ({checkpoint_path.name}):")
    for key in ("accuracy", "f1_macro", "f1_weighted"):
        print(f"  {key:12s}: {metrics[key]:.4f}")
    print(f"Artefacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
