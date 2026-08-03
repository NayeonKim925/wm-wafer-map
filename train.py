#!/usr/bin/env python3
"""Train a wafer-map defect classifier on WM-811K.

Running this is all that is required after ``pip install -r requirements.txt``:
the dataset is downloaded and verified automatically on first run.

    python train.py
    python train.py --config configs/smoke.yaml
    python train.py --set training.epochs=50 model.name=wafer_resnet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repository root importable regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cli import common_parser, create_experiment_dir, load_config_from_args  # noqa: E402
from src.config import ConfigError  # noqa: E402
from src.data import DatasetError, DatasetManager, WaferDataModule  # noqa: E402
from src.evaluation import Evaluator, save_training_curves  # noqa: E402
from src.inference import Predictor  # noqa: E402
from src.models import build_model  # noqa: E402
from src.reporting import generate_visual_report  # noqa: E402
from src.training import Trainer  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("Train a WM-811K wafer-map defect classifier.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if it already exists.",
    )
    args = parser.parse_args(argv)

    extra = ["dataset.force_download=true"] if args.force_download else []
    try:
        config = load_config_from_args(args, extra)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    run_dir = create_experiment_dir(config)
    setup_logging(args.log_level or config.log_level, log_file=run_dir / "train.log", force=True)
    logger = get_logger("train")
    logger.info("Run directory: %s", run_dir)
    config.save(run_dir / "config.resolved.yaml")

    set_seed(config.seed)
    device = resolve_device(config.training.device)

    # 1. Ensure the dataset is present (auto-download on first run).
    manager = DatasetManager(config.dataset, logger=logger)
    try:
        dataset_dir = manager.prepare()
    except DatasetError as exc:
        logger.error("Could not obtain the dataset.\n%s", exc)
        return 1

    # 2. Build the data pipeline.
    datamodule = WaferDataModule(config, dataset_dir, logger=logger)
    datamodule.setup()
    config.model.num_classes = datamodule.num_classes
    (run_dir / "label_mapping.json").write_text(
        json.dumps({"classes": datamodule.label_mapping.to_list()}, indent=2),
        encoding="utf-8",
    )

    # 3. Build the model.
    model = build_model(config.model, datamodule.input_channels, datamodule.num_classes)

    # 4. Train.
    class_weights = datamodule.class_weights() if config.training.class_weighting else None
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=run_dir,
        label_mapping=datamodule.label_mapping,
        class_weights=class_weights,
        logger=logger,
    )
    trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader())
    save_training_curves(trainer.history, run_dir / "training_curves.png")

    # 5. Evaluate the best checkpoint on the held-out test split, then build the
    #    visual experiment report.
    if config.training.evaluate_on_test:
        best_path = trainer.best_checkpoint_path
        if best_path.is_file():
            predictor = Predictor.from_checkpoint(best_path, device=device)
            evaluator = Evaluator(predictor.model, device, datamodule.label_mapping, logger=logger)
            results = evaluator.evaluate(
                datamodule.test_dataloader(), output_dir=run_dir / "evaluation"
            )
            if config.output.generate_report:
                generate_visual_report(
                    run_dir, config, datamodule, predictor, results, logger=logger
                )
        else:
            logger.warning("No best checkpoint found; skipping test evaluation.")

    logger.info("Done. All artefacts saved under %s", run_dir)
    # User-facing CLI summary.
    print(f"\nTraining complete. Best val f1_macro={trainer.best_metric:.4f}")
    print(f"Artefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
