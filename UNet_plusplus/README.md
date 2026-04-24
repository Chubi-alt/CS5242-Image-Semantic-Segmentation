## UNet++ Semantic Segmentation on CamVid
Created by: Li Junxian

---

### Overview

This repository contains the UNet++ implementation for 32-class semantic segmentation on the CamVid dataset, as part of Stage 1 parallel benchmarking. Starting from a vanilla UNet++ baseline, three progressive improvement stages are developed and evaluated across multiple backbone encoders.

**Key result:** EfficientNet-B4 V3 achieves **mIoU = 0.521** on the clear test set, a **+0.14 absolute improvement** over the EfficientNet-B3 baseline.

---

### Dataset

- **CamVid** with 32 semantic classes (31 categories + Void at index 30)
- ~367 training images; severe class imbalance
- Dominant classes: Road (28.1%), Building (23.5%), Sky (16.3%)
- Rare classes: Animal (0.005%), TrafficCone (0.003%)

---

### Optimisation Stages

**V1 — Training Strategy**
- Driving-scene data augmentation: horizontal flip, affine transforms, colour jitter, fog simulation, grid distortion, coarse dropout
- Linear warmup (10 epochs) + cosine annealing LR schedule
- Differential learning rates: encoder at `lr × 0.1`, decoder/head at full `lr`

**V2 — Loss Function**
- TripleLoss: CrossEntropy (0.2) + Dice (0.5) + Focal (0.3)
- Sqrt-smoothed median-frequency class weights, clipped at 3.0
- Label smoothing ε = 0.1

**V3 — Architecture Enhancement**
- ASPP module inserted at encoder bottleneck: parallel atrous convolutions at dilation rates {1, 6, 12, 18} + global average pooling
- GroupNorm replaces BatchNorm in global pooling branch (avoids 1×1 spatial degeneration)
- Input resolution increased from 512×512 to 640×640

---

### Results — Clear Test Set

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

### Implementation Details

| Config | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-3 |
| Base LR | 8e-4 (scaled by batch size) |
| Batch size | 12 (B3), 10 (B4), 8 (ResNet-50) |
| Max epochs | 200 with early stopping (patience = 20) |
| Mixed precision | torch.amp |
| Hardware | NVIDIA L4 24GB |

---

### Key Observations

- Training strategy (V1) contributes the largest single gain (+0.07~0.12 mIoU)
- Pretrained backbones outperform scratch by 0.08~0.10 across all versions
- ASPP + higher resolution (V3) provides the most significant architectural improvement
- EfficientNet-B4 achieves the best test generalisation despite ResNet-50 having higher validation mIoU
