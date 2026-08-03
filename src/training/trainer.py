"""Training loop with checkpointing, early stopping and metric logging."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # torch>=2.x preferred API
    from torch.amp import GradScaler, autocast

    _NEW_AMP = True
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler, autocast

    _NEW_AMP = False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a declared dependency
    tqdm = None

from src.config import Config
from src.data.labels import LabelMapping
from src.evaluation.metrics import compute_metrics
from src.training.optim import build_optimizer, build_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger


class Trainer:
    """Supervised trainer for wafer-map classification."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        output_dir: str | Path,
        label_mapping: LabelMapping,
        class_weights: torch.Tensor | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.train_cfg = config.training
        self.device = device
        self.label_mapping = label_mapping
        self.logger = logger or get_logger(__name__)

        self.model = model.to(device)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = build_optimizer(self.model.parameters(), self.train_cfg.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer, self.train_cfg.scheduler, self.train_cfg.epochs
        )

        self.use_amp = bool(self.train_cfg.mixed_precision and device.type == "cuda")
        self.scaler = self._make_scaler()

        self.history: list[dict[str, float]] = []
        self.best_metric = float("-inf")
        self._monitor_key = "f1_macro"

    # -- Public API ---------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        """Run the full training loop and return the per-epoch history."""
        patience = self.train_cfg.early_stopping_patience
        epochs_without_improvement = 0

        self.logger.info(
            "Starting training for %d epochs on %d train / %d val batches.",
            self.train_cfg.epochs,
            len(train_loader),
            len(val_loader),
        )

        for epoch in range(1, self.train_cfg.epochs + 1):
            train_stats = self._train_epoch(train_loader, epoch)
            val_stats = self._evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {"epoch": epoch, **train_stats, **_prefix(val_stats, "val_")}
            self.history.append(record)
            self._log_epoch(epoch, train_stats, val_stats)

            self._save("last.pt", epoch, val_stats)
            monitored = val_stats[self._monitor_key]
            if monitored > self.best_metric:
                self.best_metric = monitored
                epochs_without_improvement = 0
                self._save("best.pt", epoch, val_stats)
                self.logger.info(
                    "New best model (val %s=%.4f) saved.", self._monitor_key, monitored
                )
            else:
                epochs_without_improvement += 1

            self._write_history()

            if patience and epochs_without_improvement >= patience:
                self.logger.info(
                    "Early stopping at epoch %d (no val %s improvement for %d epochs).",
                    epoch,
                    self._monitor_key,
                    patience,
                )
                break

        self.logger.info("Training complete. Best val %s=%.4f", self._monitor_key, self.best_metric)
        return self.history

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    # -- Epoch loops --------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        seen = 0
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for step, (inputs, labels) in enumerate(self._progress(loader, epoch), start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.train_cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            preds.append(logits.detach().argmax(1).cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

            if step % self.train_cfg.log_interval == 0:
                self.logger.info(
                    "Epoch %d [%d/%d] loss=%.4f",
                    epoch,
                    step,
                    len(loader),
                    running_loss / max(seen, 1),
                )

        metrics = compute_metrics(np.concatenate(targets), np.concatenate(preds))
        metrics["loss"] = running_loss / max(seen, 1)
        return metrics

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        seen = 0
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for inputs, labels in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())

        metrics = compute_metrics(np.concatenate(targets), np.concatenate(preds))
        metrics["loss"] = running_loss / max(seen, 1)
        return metrics

    # -- Helpers ------------------------------------------------------------
    def _make_scaler(self):
        if _NEW_AMP:
            return GradScaler(self.device.type, enabled=self.use_amp)
        return GradScaler(enabled=self.use_amp)

    def _autocast(self):
        if _NEW_AMP:
            return autocast(self.device.type, enabled=self.use_amp)
        return autocast(enabled=self.use_amp)

    def _progress(self, loader: DataLoader, epoch: int):
        if tqdm is None:
            return loader
        return tqdm(
            loader,
            desc=f"Epoch {epoch}/{self.train_cfg.epochs}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )

    def _log_epoch(self, epoch: int, train_stats: dict, val_stats: dict) -> None:
        self.logger.info(
            "Epoch %d | train loss=%.4f acc=%.4f f1=%.4f | val loss=%.4f acc=%.4f f1=%.4f",
            epoch,
            train_stats["loss"],
            train_stats["accuracy"],
            train_stats["f1_macro"],
            val_stats["loss"],
            val_stats["accuracy"],
            val_stats["f1_macro"],
        )

    def _save(self, filename: str, epoch: int, metrics: dict[str, float]) -> None:
        save_checkpoint(
            self.checkpoint_dir / filename,
            model_state=self.model.state_dict(),
            model_config=dataclasses.asdict(self.config.model),
            label_classes=self.label_mapping.to_list(),
            epoch=epoch,
            metrics=metrics,
            config=self.config.to_dict(),
        )

    def _write_history(self) -> None:
        path = self.output_dir / "history.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def _prefix(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in stats.items()}
