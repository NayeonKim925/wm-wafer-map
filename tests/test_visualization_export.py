"""Tests for visualisations and TorchScript export."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from src.config import Config, ModelConfig
from src.data import LabelMapping, load_wm811k
from src.data.visualize import plot_class_distribution, plot_sample_wafers
from src.evaluation.plots import save_per_class_f1, save_training_curves
from src.inference import Predictor
from src.models import build_model
from src.utils.checkpoint import save_checkpoint


def test_dataset_visualisations(dataset_dir: Path, tmp_path: Path):
    mapping = LabelMapping.build()
    raw = load_wm811k(dataset_dir / "LSWMD.pkl", mapping)

    dist = plot_class_distribution(raw.labels, mapping, tmp_path / "dist.png")
    grid = plot_sample_wafers(raw.wafer_maps, raw.labels, mapping, tmp_path / "grid.png",
                              examples_per_class=3)
    assert dist.is_file()
    assert grid.is_file()


def test_training_plots(tmp_path: Path):
    history = [
        {"epoch": 1, "loss": 2.0, "f1_macro": 0.1, "val_loss": 2.1, "val_f1_macro": 0.12},
        {"epoch": 2, "loss": 1.5, "f1_macro": 0.3, "val_loss": 1.8, "val_f1_macro": 0.25},
    ]
    curves = save_training_curves(history, tmp_path / "curves.png")
    assert curves.is_file()

    report = {"Center": {"f1-score": 0.5}, "Donut": {"f1-score": 0.2}}
    bars = save_per_class_f1(report, ["Center", "Donut"], tmp_path / "f1.png")
    assert bars.is_file()


def _make_checkpoint(tmp_path: Path) -> Path:
    mapping = LabelMapping.build()
    cfg = Config.load(overrides=["data.image_size=24"])
    model = build_model(ModelConfig(base_channels=8), in_channels=3, num_classes=mapping.num_classes)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model_state=model.state_dict(),
        model_config=dataclasses.asdict(ModelConfig(base_channels=8)),
        label_classes=mapping.to_list(),
        epoch=1,
        metrics={},
        config=cfg.to_dict(),
    )
    return path


def test_torchscript_export_matches_eager(tmp_path: Path):
    ckpt = _make_checkpoint(tmp_path)
    predictor = Predictor.from_checkpoint(ckpt, device="cpu")

    ts_path = predictor.export_torchscript(tmp_path / "model.ts.pt")
    assert ts_path.is_file()

    loaded = torch.jit.load(str(ts_path))
    loaded.eval()
    x = torch.randn(2, 3, 24, 24)
    with torch.no_grad():
        expected = predictor.model(x)
        actual = loaded(x)
    assert torch.allclose(expected, actual, atol=1e-5)
