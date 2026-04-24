# CamVid Semantic Segmentation — Submission

## Best Model

**`improved_v3`** (ResNet-101 + CBAM + Dual Aux Heads + CombinedLoss + TTA)

| Metric | Value |
|--------|-------|
| Test mIoU (w/ TTA) | **0.4928** |
| Test mIoU (single-scale) | 0.4182 |
| Best Val mIoU | **0.5353** |
| Pixel Accuracy | 0.9269 |
| Checkpoint | `checkpoints/improved_v3.pt` |
| Epochs trained | 80 (early stopping, patience=15) |

---

## All Models Summary

| Model | Backbone | Best Val mIoU | Test mIoU | Checkpoint |
|-------|----------|:-------------:|:---------:|------------|
| custom_v1 | ResNet-50 | 0.4336 | 0.3914 | `checkpoints/custom.pt` |
| torchvision | ResNet-50 | 0.4799 | 0.4303 | `checkpoints/torchvision.pt` |
| improved_v1 | ResNet-50 + SE | 0.4712 | 0.4281 | `checkpoints/improved.pt` |
| custom_v2 | ResNet-101 + V3+ | 0.4928 | 0.4446 | `checkpoints/custom_v2.pt` |
| improved_v2 | ResNet-101 + CBAM | 0.5076 | 0.4473 | `checkpoints/improved_v2.pt` |
| **improved_v3** | ResNet-101 + CBAM | **0.5353** | **0.4928** | `checkpoints/improved_v3.pt` |

---

## Directory Structure

```
submission/
├── README.md                        # this file
├── checkpoints/
│   ├── improved_v3.pt               # BEST MODEL
│   ├── improved_v2.pt
│   ├── custom_v2.pt
│   ├── improved.pt
│   ├── custom.pt
│   └── torchvision.pt
├── code/
│   ├── model.py                     # v1 model (ResNet-50 DeepLabV3+)
│   ├── model_modified.py            # v2/v3 model (ResNet-101 + CBAM)
│   ├── camvid_segmentation.py       # v1 training script
│   ├── train_improved.py            # v2 training script
│   ├── train_v3.py                  # v3 training script (best)
│   ├── analyze.py                   # evaluation & report figure generation
│   ├── supplement_experiments.py    # TTA ablation, params, confusion matrix
│   ├── vis_final.py                 # prediction visualization
│   └── class_dict.csv               # CamVid class definitions (32 classes)
├── results/
│   ├── plots/                       # training curves, mIoU comparison, etc.
│   ├── tables/                      # per-class IoU CSV, model comparison CSV
│   ├── samples/                     # vis_000~007.png, failure_cases/
│   └── supplement/                  # confusion_matrix.png, TTA ablation JSON
└── outputs/
    ├── *_history.json               # per-epoch train/val metrics for each model
    └── v3_test_results.json         # best model test set per-class IoU
```

---

## Reproduction

### Environment

```bash
conda activate camvid
# PyTorch 2.5.1+cu124, Python 3.x
```

### Evaluate best model (with TTA)

```bash
python code/train_v3.py test \
    --checkpoint checkpoints/improved_v3.pt \
    --data-root /path/to/CamVid \
    --height 352 --width 480 \
    --output-stride 16
```

### Re-train best model (v3)

```bash
python code/train_v3.py train \
    --data-root /path/to/CamVid \
    --epochs 80 --batch-size 8 --lr 1e-4 \
    --height 352 --width 480 --output-stride 16 \
    --num-workers 4
```

### Reproduce all figures and tables

```bash
python code/analyze.py --data-root /path/to/CamVid
```

### TTA ablation + confusion matrix

```bash
python code/supplement_experiments.py
```

---

## TTA Ablation (improved_v3)

| Configuration | mIoU | Inference Time | Speedup |
|---------------|:----:|:--------------:|:-------:|
| Single-scale | 0.4182 | 2.1s | 1× |
| 3-scale | 0.4436 | 3.7s | 1.8× slower |
| 3-scale + flip (×6) | 0.4506 | 6.6s | 3.2× slower |

TTA (3-scale+flip) contributes **+3.24 mIoU** at 3.2× inference cost.

---

## Model Statistics (improved_v3)

| Stat | Value |
|------|-------|
| Total params | 49.3M |
| GFLOPs (352×480) | 26.3 |
| Latency (352×480) | 6.8 ms |
| Latency (448×608) | 7.1 ms |

---

## Key Design Choices (v3 vs v2)

- **Loss**: CombinedLoss = FocalLoss (γ=2) + OHEM + LabelSmoothing-CE
- **LR schedule**: PolyLR with warmup (vs CosineAnnealing in v2)
- **Input resolution**: 352×480 (same as v2)
- **Output stride**: 16
- **Early stopping**: patience=15

> Note: Epochs differ across models (32–80) due to early stopping with per-model patience settings.
