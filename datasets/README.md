# `datasets/`

This directory is the **local landing zone** for the WM-811K dataset.

**Nothing here is committed to git** (see the repository `.gitignore`). The raw
dataset is large and licensed via Kaggle, so it is downloaded automatically at
runtime instead of being stored in version control.

## How it gets populated

Running either of the following will download the dataset from Kaggle
(`qingyi/wm811k-wafer-map`) via [`kagglehub`](https://github.com/Kaggle/kagglehub),
verify it, and link it here:

```bash
python scripts/download_dataset.py     # explicit, standalone
python train.py                        # implicit, on first run
```

After a successful download you should see:

```
datasets/
└── wm811k/
    └── LSWMD.pkl        # symlink (or copy) into the kagglehub cache
```

The download location, dataset name, and expected files are all configurable in
[`configs/default.yaml`](../configs/default.yaml) under the `dataset:` section.
