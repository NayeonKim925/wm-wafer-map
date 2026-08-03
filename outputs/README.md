# `outputs/`

Runtime artefacts produced by training, evaluation, and inference land here.
**Nothing in this directory is committed to git** (see `.gitignore`).

A typical run creates a timestamped experiment directory:

```
outputs/
└── wafer_cnn/
    └── 20260803-101500/
        ├── config.resolved.yaml     # the exact config used for the run
        ├── train.log                # full training log
        ├── checkpoints/
        │   ├── best.pt              # best checkpoint (by val macro-F1)
        │   └── last.pt              # most recent checkpoint
        ├── history.json             # per-epoch metrics
        ├── label_mapping.json       # class index <-> name mapping
        └── evaluation/
            ├── classification_report.json
            ├── confusion_matrix.csv
            └── confusion_matrix.png
```

The root directory and experiment name are configurable in
[`configs/default.yaml`](../configs/default.yaml) under the `output:` section.
