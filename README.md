# WM-811K Wafer-Map Defect Classifier

A production-quality, end-to-end deep-learning pipeline for classifying
semiconductor **wafer-map defect patterns** on the
[WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).

The headline feature: **you never handle the dataset manually.** Clone, install,
and train — the dataset is downloaded, verified, and wired into the pipeline
automatically on first run.

```bash
git clone <this-repo> && cd wm-wafer-map
pip install -r requirements.txt
python train.py
```

That's it. No manual downloads, no moving files, no data committed to git.

---

## Table of contents

- [Highlights](#highlights)
- [Quickstart](#quickstart)
- [Automatic dataset download](#automatic-dataset-download)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [The dataset](#the-dataset)
- [Architecture & design](#architecture--design)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Highlights

- **Zero-touch data pipeline** — `kagglehub`-based download with integrity
  verification, managed by a reusable `DatasetManager`. The raw dataset is
  **never** committed to the repository.
- **One command to train** — `python train.py` runs download → clean → split →
  train → evaluate, end to end.
- **Leakage-free evaluation** — group-aware splitting by `lotName` (or the
  dataset's official split), so correlated wafers from one lot never straddle
  train and test. This is the difference between honest and inflated numbers.
- **Fast, correct data loading** — variable-sized wafer maps are resized **once**
  into a compact `uint8` cache (PIL is off the per-epoch hot path), and
  augmentation is seeded per DataLoader worker (avoids the classic NumPy-in-
  workers duplicate-augmentation bug).
- **Fully configurable, no hardcoded paths** — a single typed YAML config drives
  everything; every path is resolved with `pathlib` and overridable on the CLI.
- **Clean, modular architecture** — separate, single-responsibility packages for
  data, models, training, evaluation and inference; new models plug in via a
  registry.
- **Imbalance-aware metrics** — balanced accuracy, macro-F1 and Cohen's κ, not
  just accuracy (WM-811K is ~85% defect-free).
- **Visualisations** — dataset EDA (class balance + sample wafer grids),
  training curves, per-class F1, and a confusion matrix.
- **Deployment-ready** — one-command TorchScript export with a preprocessing
  sidecar, loadable by LibTorch/TorchServe with none of this codebase.
- **Proper logging** — the `logging` module throughout; `print()` is reserved
  for user-facing CLI output only.
- **Reproducible** — global + per-worker seeding, and self-describing
  checkpoints that carry their own config and label mapping.
- **Tested** — a `pytest` suite (70+ tests) validates the whole pipeline on
  synthetic data (no 2 GB download) with `kagglehub` mocked, including a
  no-lot-leakage assertion.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download just the dataset, standalone
python scripts/download_dataset.py

# 3. Train (auto-downloads the dataset if step 2 was skipped)
python train.py

# 4. Evaluate a trained checkpoint on the test split
python evaluate.py --checkpoint outputs/wafer_cnn/<run>/checkpoints/best.pt

# 5. Classify new wafer maps
python inference.py --checkpoint outputs/wafer_cnn/<run>/checkpoints/best.pt --input wafers.npy
```

Want to verify everything works in ~1 minute on a CPU first? Use the smoke
config, which trains a small model on a capped subset:

```bash
python train.py --config configs/smoke.yaml
```

## Automatic dataset download

The dataset is fetched on demand and **never stored in version control**
(`datasets/` and all `*.pkl` files are git-ignored).

**How it works.** Both `python train.py` and `python scripts/download_dataset.py`
delegate to `DatasetManager` (`src/data/dataset_manager.py`), which:

1. **Locates** an existing copy under `datasets/wm811k/`.
2. **Downloads** from Kaggle via `kagglehub.dataset_download("qingyi/wm811k-wafer-map")`
   if it is missing.
3. **Materialises** the file into the project-local `datasets/` directory (a
   symlink by default — no 2 GB copy — with an automatic copy fallback).
4. **Verifies** that the expected files exist and are not truncated.
5. **Returns** a stable path the rest of the pipeline consumes.

Re-runs are cheap: an existing, verified copy is detected and the download is
skipped.

```bash
# Standalone downloader
python scripts/download_dataset.py          # download + verify + print location
python scripts/download_dataset.py --force  # force a fresh re-download
python scripts/download_dataset.py --set dataset.link_mode=copy   # copy instead of symlink
```

**Kaggle credentials.** `qingyi/wm811k-wafer-map` is public and usually
downloads anonymously. If you hit an authentication/403 error, provide
credentials (see [`.env.example`](.env.example)):

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
# or place a token at ~/.kaggle/kaggle.json (chmod 600)
```

## Project structure

```
wm-wafer-map/
├── train.py                 # Entry point: end-to-end training (auto-downloads data)
├── evaluate.py              # Entry point: evaluate a checkpoint on a split
├── inference.py             # Entry point: classify new wafer maps
├── requirements.txt         # Runtime dependencies
├── pyproject.toml           # Packaging + tooling (ruff, pytest) config
│
├── configs/                 # YAML configuration (no paths hardcoded in code)
│   ├── default.yaml         # Canonical, fully-documented config
│   └── smoke.yaml           # Fast CPU smoke-test overrides
│
├── scripts/
│   ├── download_dataset.py  # Standalone dataset downloader/verifier
│   ├── visualize_dataset.py # Dataset EDA (class balance + sample wafers)
│   └── export_model.py      # Export a checkpoint to TorchScript for serving
│
├── src/                     # Importable source package
│   ├── cli.py               # Shared CLI helpers (arg parsing, run dirs)
│   ├── config/              # Typed, hierarchical config (dataclasses + YAML)
│   ├── data/                # Dataset acquisition, cleaning, batching
│   │   ├── dataset_manager.py   # DatasetManager: locate/download/verify
│   │   ├── dataset.py           # WaferMapDataset + WM-811K loader + augment
│   │   ├── datamodule.py        # Leakage-free train/val/test dataloaders
│   │   ├── preprocessing.py     # Resize/encode wafer maps, clean labels
│   │   ├── visualize.py         # Class-distribution + sample-wafer figures
│   │   └── labels.py            # Class definitions + LabelMapping
│   ├── models/              # Architectures + build registry
│   │   ├── cnn.py               # WaferCNN
│   │   ├── resnet.py            # WaferResNet
│   │   ├── blocks.py            # Reusable conv/residual blocks
│   │   └── factory.py           # register_model / build_model
│   ├── training/            # Trainer, optimizers, schedulers
│   ├── evaluation/          # Metrics, evaluator, plots
│   ├── inference/           # Predictor + TorchScript export
│   └── utils/               # Logging, seeding, device, paths, plotting
│
├── datasets/                # Auto-populated, git-ignored (data lives here)
├── outputs/                 # Auto-populated, git-ignored (checkpoints, logs)
└── tests/                   # pytest suite (runs on synthetic data)
```

> **Note on `models/` and `utils/`.** Following the standard Python *src-layout*,
> model architectures live in `src/models/` and shared utilities in `src/utils/`,
> so the whole project is one clean, importable package.

## Configuration

Everything is driven by [`configs/default.yaml`](configs/default.yaml) — a single,
fully-commented source of truth. There are three ways to set values, in
increasing precedence:

1. Dataclass defaults in `src/config/config.py` (the safety net).
2. A YAML file (`--config path/to/config.yaml`).
3. Dotted CLI overrides (`--set key.path=value`).

```bash
# Use an alternative config file
python train.py --config configs/smoke.yaml

# Override individual values (parsed as typed YAML scalars)
python train.py --set training.epochs=50 data.batch_size=256 model.name=wafer_resnet

# Combine both
python train.py --config configs/smoke.yaml --set training.optimizer.lr=0.0005
```

Unknown keys are rejected with a helpful error, so typos never silently
misconfigure a run. The exact, resolved config for every run is written to
`outputs/<experiment>/<run>/config.resolved.yaml` for reproducibility.

## Training

```bash
python train.py                                   # full run with defaults
python train.py --config configs/smoke.yaml       # quick CPU sanity run
python train.py --set training.epochs=60          # longer training
python train.py --set model.name=wafer_resnet     # switch architecture
python train.py --force-download                  # refetch the dataset first
```

Each run creates a timestamped directory under `outputs/` containing:

```
outputs/wafer_cnn/<timestamp>/
├── config.resolved.yaml        # exact config used
├── train.log                   # full training log
├── label_mapping.json          # class index <-> name mapping
├── history.json                # per-epoch metrics
├── training_curves.png         # loss & macro-F1 vs epoch (train/val)
├── checkpoints/
│   ├── best.pt                 # best by validation macro-F1
│   └── last.pt                 # most recent epoch
└── evaluation/                 # test-split metrics (if enabled)
    ├── classification_report.json
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── per_class_f1.png
```

**What training does:** resolves the device (CUDA → MPS → CPU), auto-downloads
the data, builds a **leakage-free split** (see below), caches resized wafer
maps, trains with a class-weighted loss (WM-811K is heavily imbalanced), cosine
LR schedule, gradient clipping, optional mixed precision, early stopping on
validation macro-F1, checkpointing, and a final evaluation of the best model on
the held-out test split.

## Data splitting & evaluation protocol

How the data is split is the single biggest driver of whether reported numbers
are honest. WM-811K wafers are grouped into **lots** processed under shared
conditions, so wafers from one lot are strongly correlated. Splitting them
randomly leaks information from test into train and **inflates** results. This
project defaults to a leakage-free protocol (`data.split_strategy`):

| Strategy | Leakage-free | Use it for |
| --- | --- | --- |
| `lot` (default) | ✅ no lot spans two splits | Honest generalisation to unseen lots |
| `official` | ✅ train⊥val by lot; test = dataset's own `trianTestLabel` | Comparability with published WM-811K benchmarks¹ |
| `random` | ❌ | Quick experiments only — expect optimistic metrics |

```bash
python train.py --set data.split_strategy=lot        # default, leakage-free
python train.py --set data.split_strategy=official   # dataset's own train/test
python train.py --set data.split_strategy=random     # fast but biased
```

¹ The dataset's official `Training`/`Test` flag is assigned per wafer and is not
guaranteed lot-disjoint, so `official` can still share lots between train and
test — that's a property of the dataset, and precisely why `lot` exists.

For a rigorous result, run several seeds and report mean ± std
(`--set seed=0`, `seed=1`, …); the split, weight init and augmentation all key
off the global seed. (k-fold cross-validation is a natural extension.)

## Dataset visualisation (EDA)

Understand the data before trusting a model:

```bash
python scripts/visualize_dataset.py                       # writes to outputs/eda/
python scripts/visualize_dataset.py --examples 8 --output-dir reports/eda
```

Produces a class-distribution chart (the imbalance is stark) and a grid of
sample wafer maps per class (what each defect pattern actually looks like).

## Evaluation

Evaluate any checkpoint on the `train`, `val`, or `test` split. The config
stored **inside** the checkpoint is reused by default, so the split and label
mapping exactly match training:

```bash
python evaluate.py --checkpoint outputs/wafer_cnn/<run>/checkpoints/best.pt
python evaluate.py --checkpoint <ckpt> --split val
python evaluate.py --checkpoint <ckpt> --output-dir reports/eval1
```

**Reported metrics** (chosen for a heavily imbalanced problem): accuracy,
**balanced accuracy** (mean per-class recall), macro/weighted precision-recall-F1,
**Cohen's κ**, a full per-class report, a confusion matrix (CSV + normalised
heatmap), and a per-class F1 bar chart.

## Deployment (TorchScript export)

Export a trained checkpoint to a self-contained TorchScript module that loads
with `torch.jit.load` — no dependency on this codebase — for LibTorch/TorchServe:

```bash
python scripts/export_model.py --checkpoint <ckpt> --output outputs/export/model.ts.pt
```

A JSON sidecar records the preprocessing contract (input channels, image size,
class order) the serving side needs. (For newer runtimes, `torch.export`/ONNX
are natural alternatives.)

## Inference

Classify new wafer maps from a `.npy` (single 2-D array or object array of them)
or a `.pkl` (a DataFrame with a `waferMap` column, or a list of maps):

```bash
python inference.py --checkpoint <ckpt> --input wafer.npy
python inference.py --checkpoint <ckpt> --input wafers.pkl --top-k 5 --output preds.json
```

Output per wafer map: the predicted class, a confidence score, and the top-k
class probabilities. The checkpoint is self-describing — it rebuilds an
identical model and label mapping with no external state.

Programmatic use:

```python
from src.inference import Predictor

predictor = Predictor.from_checkpoint("outputs/wafer_cnn/<run>/checkpoints/best.pt")
result = predictor.predict_one(wafer_map)   # wafer_map: 2-D numpy array of {0,1,2}
print(result["predicted_class"], result["confidence"])
```

## The dataset

**WM-811K** contains 811,457 real-world wafer maps; about 172,950 are
hand-labelled. Each wafer map is a variable-sized 2-D grid of
`{0: background, 1: passing die, 2: failing die}`. Labels are one of nine
classes — eight defect patterns plus defect-free:

`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`,
`Scratch`, and `none`.

The pipeline cleans the dataset's quirky nested-array labels, drops unlabelled
rows, optionally excludes the majority `none` class, resizes each map to a fixed
resolution with categorical-preserving nearest-neighbour interpolation (**once**,
into a cached `uint8` stack), and encodes it as either a single scaled channel or
a 3-channel one-hot tensor. The `lotName` and official train/test flag are
retained so splitting can be leakage-free (see above).

## Architecture & design

The codebase follows clean-architecture principles — small, focused modules with
one responsibility each and no duplicated logic:

- **Config-driven, path-agnostic.** No path is hardcoded; all are configured and
  resolved via `pathlib`.
- **`DatasetManager`** owns data acquisition and integrity — nothing else knows
  how the data got there.
- **`WaferDataModule`** owns loading/cleaning/splitting/batching and exposes
  dataloaders + class weights. Splitting is pluggable (`lot` / `official` /
  `random`) and leakage-free by default.
- **Model registry.** Add an architecture by decorating a builder with
  `@register_model("name")`; `build_model` and the config pick it up with no
  other changes.
- **`Trainer` / `Evaluator` / `Predictor`** are cleanly separated; metrics and
  plotting helpers live in one place and are shared (no duplicated logic).
- **Self-describing checkpoints** carry weights + model config + label mapping +
  run config, so evaluation, inference and export need nothing external.

Extending it is easy: register a new model in `src/models/`, add a new data
representation in `src/data/preprocessing.py`, or point `--config` at your own
YAML.

## Testing

The suite (70+ tests) runs the entire pipeline on a small **synthetic** dataset
with the exact WM-811K schema, and mocks `kagglehub`, so no large download is
needed. It includes an explicit **no-lot-leakage** assertion for the `lot` and
`official` splits, a TorchScript round-trip check, and worker-seeding tests:

```bash
pip install -r requirements-dev.txt
pytest                 # run everything
ruff check .           # lint
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `DatasetDownloadError` / 403 from Kaggle | Set `KAGGLE_USERNAME` / `KAGGLE_KEY` or add `~/.kaggle/kaggle.json` (see [`.env.example`](.env.example)). |
| `DatasetVerificationError: undersized file` | A truncated download; re-run with `--force-download`. |
| Out of memory | Lower `data.batch_size`, `data.image_size`, or set `data.max_samples`. |
| Training is slow on CPU | Use `configs/smoke.yaml`, or run on a CUDA GPU (auto-detected). |
| Symlink errors on the dataset | `--set dataset.link_mode=copy`. |

## License

Released under the [MIT License](LICENSE). The WM-811K dataset is distributed by
its authors via Kaggle under its own terms.
