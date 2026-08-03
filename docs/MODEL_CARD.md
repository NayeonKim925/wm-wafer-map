# Model Card — WM-811K Wafer-Map Defect Classifier

This card follows the spirit of [Mitchell et al., 2019, *Model Cards for Model
Reporting*].

## Model details
- **Task:** multi-class classification of wafer-map defect patterns.
- **Architectures:** `wafer_cnn` (a compact 4-stage CNN with global average
  pooling) or `wafer_resnet` (small residual network), selectable via config.
- **Input:** a wafer map — a 2-D grid of `{0: background, 1: passing die,
  2: failing die}` — resized (nearest-neighbour) to a fixed resolution and
  encoded as a 3-channel one-hot tensor (default) or a single scaled channel.
- **Output:** a probability distribution over 9 classes (`Center`, `Donut`,
  `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`, `none`).
- **Framework:** PyTorch; exportable to TorchScript for serving.

## Intended use
- **Primary:** research and education on semiconductor defect-pattern
  recognition; a reference implementation of an end-to-end CV pipeline.
- **Out of scope:** production yield decisions without domain validation, and
  any use on wafer data whose acquisition differs materially from WM-811K.

## Training & evaluation data
- **Dataset:** [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
  — 811,457 real wafer maps, ~172,950 hand-labelled. Downloaded automatically at
  runtime; never committed to the repository.
- **Splitting:** **leakage-free by default** — a group-aware split by `lotName`
  ensures wafers from one lot never appear in more than one split. The dataset's
  official `trianTestLabel` split is also supported for benchmark comparability.
  A plain random split is available but is known to inflate metrics.

## Metrics
Because the labelled subset is dominated by the `none` class (~85%), the model is
evaluated with imbalance-aware metrics, not accuracy alone:
- **Balanced accuracy** (mean per-class recall), **macro-F1**, **Cohen's κ**,
  a full per-class precision/recall/F1 report, a normalised confusion matrix, and
  a calibration (reliability) diagram with Expected Calibration Error.

Run `python train.py` to (re)produce these on your hardware; every run writes a
self-contained `report.md` with the numbers and figures.

## Ethical considerations & limitations
- **Class imbalance:** rare defect types have few examples; per-class recall
  should be inspected, not just aggregate accuracy.
- **Domain shift:** models trained on WM-811K may not transfer to wafers from
  other fabs, sizes, or process nodes without re-training.
- **Label noise:** WM-811K labels are human-assigned and imperfect.
- **Explainability:** Grad-CAM is provided as a sanity check that predictions are
  driven by the actual defect signature rather than artefacts — but it is an
  approximation, not a guarantee of correctness.

## How to use
```python
from src.inference import Predictor

predictor = Predictor.from_checkpoint("outputs/wafer_cnn/<run>/checkpoints/best.pt")
result = predictor.predict_one(wafer_map)   # wafer_map: 2-D numpy array of {0,1,2}
print(result["predicted_class"], result["confidence"])
```
