"""Integration tests: trainer -> evaluator -> predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import WaferDataModule
from src.evaluation import Evaluator
from src.inference import Predictor
from src.models import build_model
from src.training import Trainer
from src.utils.device import resolve_device


def _prepare(make_config, dataset_dir: Path, out_dir: Path):
    cfg = make_config(dataset_dir.parent)
    device = resolve_device("cpu")
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
    cfg.model.num_classes = dm.num_classes
    model = build_model(cfg.model, dm.input_channels, dm.num_classes)
    trainer = Trainer(
        model, cfg, device, out_dir, dm.label_mapping, class_weights=dm.class_weights()
    )
    return cfg, device, dm, model, trainer


def test_trainer_produces_checkpoints_and_history(make_config, dataset_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "run"
    _, _, dm, _, trainer = _prepare(make_config, dataset_dir, out_dir)
    history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    assert len(history) == 1
    assert (out_dir / "checkpoints" / "best.pt").is_file()
    assert (out_dir / "checkpoints" / "last.pt").is_file()
    assert (out_dir / "history.json").is_file()
    assert "val_f1_macro" in history[0]


def test_evaluator_writes_artefacts(make_config, dataset_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "run"
    _, device, dm, model, trainer = _prepare(make_config, dataset_dir, out_dir)
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    evaluator = Evaluator(model, device, dm.label_mapping)
    results = evaluator.evaluate(dm.test_dataloader(), output_dir=out_dir / "evaluation")

    assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
    assert (out_dir / "evaluation" / "classification_report.json").is_file()
    assert (out_dir / "evaluation" / "confusion_matrix.csv").is_file()
    cm = results["confusion_matrix"]
    assert cm.shape == (dm.num_classes, dm.num_classes)


def test_predictor_from_checkpoint(make_config, dataset_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "run"
    _, _, dm, _, trainer = _prepare(make_config, dataset_dir, out_dir)
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    predictor = Predictor.from_checkpoint(out_dir / "checkpoints" / "best.pt", device="cpu")
    wafer = dm._wafer_maps[0]

    single = predictor.predict_one(wafer, top_k=3)
    assert single["predicted_class"] in dm.label_mapping.classes
    assert 0.0 <= single["confidence"] <= 1.0
    assert len(single["top_k"]) == 3

    batch = predictor.predict([wafer, dm._wafer_maps[1]])
    assert len(batch) == 2


def test_predictor_probabilities_sum_to_one(make_config, dataset_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "run"
    _, _, dm, _, trainer = _prepare(make_config, dataset_dir, out_dir)
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())
    predictor = Predictor.from_checkpoint(out_dir / "checkpoints" / "best.pt", device="cpu")

    # Reconstruct full distribution via a large top_k.
    result = predictor.predict_one(dm._wafer_maps[0], top_k=dm.num_classes)
    total = sum(item["probability"] for item in result["top_k"])
    assert total == np.float32(1.0) or abs(total - 1.0) < 1e-4
