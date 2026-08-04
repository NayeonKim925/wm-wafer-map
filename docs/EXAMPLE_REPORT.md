# Experiment report — wafer_cnn

_Auto-generated 2026-08-04 00:30:24._

## Run configuration

| Setting | Value |
| --- | --- |
| Model | `wafer_cnn` (base_channels=32) |
| Split strategy | `lot` |
| Image size | 64 |
| Representation | `onehot` |
| Epochs | 30 |
| Optimizer | `adamw` (lr=0.001) |
| Classes | 9 |
| Seed | 42 |

## Test metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.9614 |
| Balanced accuracy | 0.9019 |
| Macro-F1 | 0.8370 |
| Weighted-F1 | 0.9641 |
| Cohen's κ | 0.8626 |

## Per-class F1 (test)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Center | 0.833 | 0.927 | 0.877 | 698 |
| Donut | 0.805 | 0.892 | 0.847 | 102 |
| Edge-Loc | 0.680 | 0.889 | 0.771 | 728 |
| Edge-Ring | 0.979 | 0.979 | 0.979 | 1337 |
| Loc | 0.674 | 0.754 | 0.711 | 548 |
| Near-full | 0.909 | 0.833 | 0.870 | 24 |
| Random | 0.833 | 0.960 | 0.892 | 125 |
| Scratch | 0.451 | 0.914 | 0.604 | 162 |
| none | 0.995 | 0.970 | 0.982 | 22100 |

## Figures

**Training and validation loss / macro-F1 per epoch.**

![training_curves](assets/training_curves.png)

**Row-normalised confusion matrix on the test split.**

![confusion_matrix](assets/confusion_matrix.png)

**Per-class F1 on the test split.**

![per_class_f1](assets/per_class_f1.png)

**Sample predictions with class probabilities.**

![prediction_gallery](assets/prediction_gallery.png)

**Grad-CAM attention for the predicted class.**

![gradcam](assets/gradcam.png)

**Error analysis: misclassified wafers (true vs predicted).**

![misclassified](assets/misclassified.png)

**Calibration (reliability) diagram with ECE.**

![reliability](assets/reliability.png)
