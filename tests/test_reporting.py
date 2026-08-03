"""Tests for prediction/Grad-CAM galleries, error analysis, and the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import WaferDataModule
from src.evaluation import Evaluator, save_misclassified_grid, save_reliability_diagram
from src.inference import Predictor
from src.inference.visualize import plot_prediction_gallery
from src.interpretability import plot_gradcam_gallery
from src.models import build_model
from src.reporting import generate_visual_report
from src.training import Trainer
from src.utils.device import resolve_device


def _train_small(make_config, dataset_dir: Path, out_dir: Path):
    cfg = make_config(dataset_dir.parent)
    device = resolve_device("cpu")
    dm = WaferDataModule(cfg, dataset_dir)
    dm.setup()
    cfg.model.num_classes = dm.num_classes
    model = build_model(cfg.model, dm.input_channels, dm.num_classes)
    trainer = Trainer(model, cfg, device, out_dir, dm.label_mapping, class_weights=dm.class_weights())
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())
    predictor = Predictor.from_checkpoint(out_dir / "checkpoints" / "best.pt", device="cpu")
    return cfg, device, dm, predictor


def test_prediction_and_gradcam_galleries(make_config, dataset_dir: Path, tmp_path: Path):
    _, _, dm, predictor = _train_small(make_config, dataset_dir, tmp_path / "run")
    wafers = dm.wafer_maps[:6]

    gallery = plot_prediction_gallery(predictor, wafers, tmp_path / "gallery.png", max_samples=6)
    gradcam = plot_gradcam_gallery(predictor, wafers, tmp_path / "gradcam.png", max_samples=6)
    assert gallery.is_file()
    assert gradcam.is_file()


def test_error_and_reliability_plots(make_config, dataset_dir: Path, tmp_path: Path):
    _, device, dm, predictor = _train_small(make_config, dataset_dir, tmp_path / "run")
    evaluator = Evaluator(predictor.model, device, dm.label_mapping)
    results = evaluator.evaluate(dm.test_dataloader())

    test_wafers = [dm.wafer_maps[int(i)] for i in dm.test_dataset.indices]
    reliability = save_reliability_diagram(
        results["y_true"], results["y_prob"], tmp_path / "rel.png"
    )
    assert reliability.is_file()

    # Force at least one "misclassification" so the montage renders deterministically.
    y_true = results["y_true"].copy()
    y_pred = results["y_pred"].copy()
    if np.array_equal(y_true, y_pred):
        y_pred[0] = (y_pred[0] + 1) % dm.num_classes
    grid = save_misclassified_grid(test_wafers, y_true, y_pred, dm.label_mapping, tmp_path / "err.png")
    assert grid is not None and grid.is_file()


def test_generate_visual_report(make_config, dataset_dir: Path, tmp_path: Path):
    run_dir = tmp_path / "run"
    _, device, dm, predictor = _train_small(make_config, dataset_dir, run_dir)
    evaluator = Evaluator(predictor.model, device, dm.label_mapping)
    results = evaluator.evaluate(dm.test_dataloader(), output_dir=run_dir / "evaluation")

    report_path = generate_visual_report(run_dir, make_config(dataset_dir.parent), dm, predictor, results)
    assert report_path.is_file()
    text = report_path.read_text()
    assert "# Experiment report" in text
    assert "Test metrics" in text
    # Galleries were produced.
    assert (run_dir / "report" / "prediction_gallery.png").is_file()
    assert (run_dir / "report" / "gradcam.png").is_file()
