# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Grad-CAM explainability (`src/interpretability/`) with an attention-overlay
  gallery, exposed via `inference.py --gradcam`.
- Inference prediction gallery (wafer + top-k probabilities) via
  `inference.py --gallery`.
- Error-analysis montage of misclassified wafers and a calibration (reliability)
  diagram with Expected Calibration Error.
- Automatic Markdown experiment report (`report.md`) assembling metrics and every
  figure, generated after training.
- Dataset EDA script (`scripts/visualize_dataset.py`): class balance + sample
  wafer grids.
- TorchScript export (`scripts/export_model.py`) with a preprocessing sidecar.
- Continuous integration (GitHub Actions): ruff + pytest on Python 3.10–3.12.

### Changed
- **Leakage-free data splitting** by `lotName` (default) or the dataset's
  official `trianTestLabel`, replacing the earlier random split.
- Wafer maps are resized once into a cached `uint8` stack (PIL off the per-epoch
  hot path); augmentation is seeded per DataLoader worker.
- Metrics now include balanced accuracy and Cohen's κ for the imbalanced setting.

## [0.1.0]

### Added
- Initial production-quality pipeline: automated `kagglehub` dataset download and
  verification (`DatasetManager`), typed YAML configuration, PyTorch data module,
  CNN/ResNet architectures behind a build registry, trainer with class-weighted
  loss and early stopping, evaluator, predictor, and a synthetic-data test suite.
