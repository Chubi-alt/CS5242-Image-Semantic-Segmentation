# UNet++ Semantic Segmentation on CamVid

UNet++ with ASPP for 32-class semantic segmentation on the CamVid dataset.
Best result: **EfficientNet-B4 V3**, test mIoU = **0.521** on the clear test set.

---

## Repository Structure

```
UNet_plusplus/
├── models/
│   ├── builder.py              # Model factory (SMP + scratch)
│   └── my_unetpp.py            # Hand-rolled UNet++ implementation
├── utils/
│   ├── dataset.py              # CamVidDataset
│   ├── helpers.py              # Experiment utilities
│   └── visualizer.py           # Plotting and visualisation
├── evaluation_matrix/
│   ├── miou.py
│   ├── pixel_accuracy.py
│   ├── dice_coefficient.py
│   ├── fwiou.py
│   └── boundary_iou.py
├── scripts/
│   └── outputs/
│       └── class_stats.csv     # Pre-computed pixel frequency stats
├── train_aspp.py               # Training script (EfficientNet-B3/B4, V3)
├── train_aspp_resnet50.py      # Training script (ResNet-50, V3)
├── train_weighted_loss.py      # Training script (V1/V2 baselines)
├── test_and_vis.py             # Evaluation and visualisation
└── fill_dice.py                # Utility: fill missing val_dice in CSV
```

---

## Environment Setup

```bash
pip install torch torchvision segmentation-models-pytorch albumentations \
            opencv-python-headless pandas numpy matplotlib tqdm
```

Tested on Python 3.10, PyTorch 2.0, NVIDIA L4 24GB.

---

## Dataset

Download the CamVid dataset and place it under `../data/CamVid/` relative to this folder:

```
data/CamVid/
├── train/                  # Training images
├── train_labels_indexed/   # Indexed segmentation masks (0-31)
├── val/
├── val_labels_indexed/
├── test/
├── test_labels_indexed/
└── class_dict.csv          # Class name and RGB colour mapping
```

The indexed masks use integer values 0–31 corresponding to rows in `class_dict.csv`. Class 30 (Void, RGB `[0,0,0]`) is excluded from all loss computations and evaluation metrics.

---

## Training

### Improved V3 — EfficientNet-B3 / B4 (ASPP + 640px)

```bash
python train_aspp.py
```

Key configuration (edit at the top of `train_aspp.py`):

| Parameter | Value |
|---|---|
| `BACKBONE` | `efficientnet-b3` or `efficientnet-b4` |
| `ENCODER_LAST_CHANNELS` | 384 (B3) / 448 (B4) |
| `BATCH_SIZE` | 12 (B3) / 10 (B4) |
| `IMG_SIZE` | 640 |
| `EPOCHS` | 200 |
| `PATIENCE` | 20 |

### Improved V3 — ResNet-50 (ASPP + 640px)

```bash
python train_aspp_resnet50.py
```

Key configuration:

| Parameter | Value |
|---|---|
| `BACKBONE` | `resnet50` |
| `ENCODER_LAST_CHANNELS` | 2048 |
| `BATCH_SIZE` | 8 |
| `IMG_SIZE` | 640 |

### Improved V1 / V2 Baselines

```bash
python train_weighted_loss.py
```

Set `BACKBONE` to `resnet34`, `resnet50`, `efficientnet-b3`, or `scratch`.

### Resuming Training

Set `RESUME_CHECKPOINT` to the path of an existing `best.pth`:

```python
RESUME_CHECKPOINT = "checkpoints/your_experiment_name/best.pth"
```

The script automatically restores training history, best mIoU, and learning rate scheduler state.

---

## Evaluation

```bash
python test_and_vis.py
```

Edit the configuration at the top of `test_and_vis.py`:

```python
EXPERIMENT_NAME       = "unetpp_efficientnet-b4_improved_v3_XXXXXXXX_XXXX"
BACKBONE              = "efficientnet-b4"
ENCODER_LAST_CHANNELS = 448
USE_ASPP              = True
IMG_SIZE              = 640
WEATHER               = "clear"   # or "rainy"
```

Outputs are saved to `outputs/{EXPERIMENT_NAME}/test_results/`:
- `performance_on_{WEATHER}.txt` — quantitative metrics
- `test_prediction_on_{WEATHER}_*.png` — qualitative visualisations

---

## Results — Clear Test Set

| Backbone | Version | mIoU | Pixel Acc | Mean Dice | FWIoU | Boundary IoU |
|---|---|---|---|---|---|---|
| ResNet-34 | Baseline | 0.3759 | 0.8856 | 0.4740 | 0.8307 | 0.3759 |
| ResNet-34 | V1 | 0.4499 | 0.8986 | 0.5689 | 0.8505 | 0.4498 |
| ResNet-50 | Baseline | 0.3924 | 0.8915 | 0.4925 | 0.8377 | 0.3924 |
| ResNet-50 | V1 | 0.4706 | 0.9021 | 0.5960 | 0.8534 | 0.4705 |
| ResNet-50 | V2 | 0.4827 | 0.9019 | 0.6049 | 0.8540 | 0.4825 |
| ResNet-50 | V3 | 0.5119 | 0.8799 | 0.5921 | 0.8529 | 0.5118 |
| EfficientNet-B3 | Baseline | 0.3594 | 0.8856 | 0.4534 | 0.8321 | 0.3595 |
| EfficientNet-B3 | V1 | 0.4769 | 0.9036 | 0.5961 | 0.8570 | 0.4768 |
| EfficientNet-B3 | V2 | 0.4951 | 0.8768 | 0.5761 | 0.8440 | 0.4951 |
| EfficientNet-B3 | V3 | 0.5134 | 0.8828 | 0.5859 | 0.8539 | 0.5133 |
| **EfficientNet-B4** | **V3** | **0.5210** | **0.8874** | **0.5966** | **0.8612** | **0.5210** |
| Scratch | Baseline | 0.3639 | 0.8645 | 0.4739 | 0.8037 | 0.3636 |
| Scratch | V1 | 0.4037 | 0.8794 | 0.5255 | 0.8209 | 0.4034 |
| Scratch | V2 | 0.4426 | 0.8656 | 0.5300 | 0.8250 | 0.4421 |

---

## Optimisation Summary

| Version | Key Changes | mIoU Gain |
|---|---|---|
| Baseline | Standard UNet++, CrossEntropy, fixed LR | — |
| V1 | Augmentation, warmup + cosine LR, differential LR | +0.07~0.12 |
| V2 | TripleLoss (CE+Dice+Focal), sqrt class weights | +0.01~0.02 |
| V3 | ASPP at encoder bottleneck, 512→640 resolution | +0.02~0.04 |

---

## Checkpoint Format

V3 checkpoints (with ASPP) are saved as a dict:

```python
{
    "model_state_dict": model.state_dict(),
    "aspp_state_dict":  aspp.state_dict(),
}
```

V1/V2 checkpoints are plain `model.state_dict()`.

---

## Hardware

Trained on NVIDIA L4 24GB. Approximate training time per experiment: 3–6 hours.
