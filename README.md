# Enhancing Vision-Language Spatial Reasoning through Mask-Guided Semantic Segmentation

A two-stage framework that investigates whether semantic segmentation can serve as a structured visual prior to improve spatial grounding in Vision-Language Model (VLM)-based scene understanding, evaluated on the CamVid urban driving dataset.

---

## Overview

Current Vision-Language Models frequently exhibit spatial hallucinations and unreliable object localization in complex driving scenes. This project addresses this limitation through a decoupled pipeline:

- **Stage 1** — Systematic benchmarking of multiple segmentation architectures on the CamVid 32-class dataset under a unified progressive training protocol (V1→V2→V3).
- **Stage 2** — Using the best segmentation model to generate semantic masks that are supplied alongside the original RGB frame to GPT-4o, comparing RGB-only vs. segmentation-guided scene descriptions.

---

## Repository Structure

```
.
├── UNet_baseline/              # Classical UNet trained from scratch
│   ├── dataset.py
│   ├── train.py
│   ├── test.py
│   └── unet_model.py
├── UNet_plusplus/              # UNet++ with pretrained backbones and ASPP
│   ├── models/
│   │   ├── builder.py
│   │   └── my_unetpp.py
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── helpers.py
│   │   └── visualizer.py
│   ├── train_baseline.py
│   ├── train_aspp.py
│   ├── train_weighted_loss.py
│   └── train_transform.py
├── segformer_training/         # SegFormer-B1 with progressive recipe ladder
│   ├── configs/                # YAML configs for all experiments
│   ├── core/                   # Dataset, model, trainer, losses, metrics, etc.
│   └── scripts/
│       └── run.py
├── VLM_generation/             # Stage 2: mask-guided VLM reasoning
│   ├── main.py
│   ├── mask_guided_isolation.py
│   ├── vlm_api_reasoning.py
│   ├── vlm_with_mask.py
│   ├── evaluate_vlm_descriptions.py
│   ├── ground_truth_annotations.json
│   └── vlm_evaluation_results.json
├── yolosam_scripts_data/       # YOLO-SAM hybrid experiments and visualizations
├── evaluation_matrix/          # Evaluation metric implementations
│   ├── miou.py
│   ├── dice_coefficient.py
│   ├── pixel_accuracy.py
│   ├── fwiou.py
│   ├── boundary_iou.py
│   └── evaluate.py
├── scripts/                    # Data preprocessing and visualization
│   ├── preprocess_masks.py
│   └── visualize_class_imbalance.py
└── data -> /path/to/camvid     # Symlink to CamVid dataset (not included)
```

---

## Dataset

**CamVid** (Cambridge-driving Labeled Video Database) — a road scene understanding dataset with densely annotated frames at 960×720 resolution.

- **32-class** variant
- Split: 367 train / 101 val / 233 test images
- Severe long-tail class imbalance: Road + Building + Sky account for ~68% of all pixels, while rare classes (Animal, TrafficCone) occupy less than 0.01%

> The dataset is not included in this repository. Place it at the path pointed to by the `data` symlink, or update the symlink accordingly.

---

## Models

### UNet Baseline
Classical encoder-decoder architecture trained from scratch. Four training recipes are evaluated (Baseline, Boundary-weighted loss, Color augmentation, Multi-resolution), isolating the effect of optimization choices.

| Recipe    | mIoU  | Pixel Acc. |
|-----------|-------|------------|
| Baseline  | 0.380 | 0.873      |
| Boundary  | 0.395 | 0.879      |
| Color     | 0.370 | 0.876      |
| MultiRes  | 0.249 | 0.770      |

### UNet++
Nested dense skip pathways with pretrained backbones (ResNet-34/50, EfficientNet-B3/B4) and an ASPP bottleneck module. Best result: **EfficientNet-B4, V3 → mIoU = 0.521**.

### DeepLabV3+
Custom implementation with ResNet-50/101 backbones, ASPP, CBAM attention, and auxiliary heads. Progressive training recipe (CutMix, OHEM, Focal loss, TTA) brings V3 to **mIoU = 0.493**.

### SegFormer-B1
Hierarchical transformer encoder (MiT-B1) with an all-MLP decoder. ClassMix augmentation and Lovász-Softmax loss address long-tail collapse. V3 reaches **mIoU = 0.496**.

### Hybrid YOLO-SAM
YOLOv8s / YOLO26s provides bounding-box prompts to SAM (ViT-B) for mask generation. End-to-end YOLO26s-seg outperforms the hybrid pipeline at **mIoU = 0.441**.

---

## Training

Each model family has its own training script and configuration. Example for SegFormer:

```bash
cd segformer_training
python scripts/run.py --config configs/baseline.yaml
# or for the best V3 configuration:
python scripts/run.py --config configs/exp_all.yaml
```

For UNet++:
```bash
cd UNet_plusplus
python train_aspp.py          # V3: ASPP + TripleLoss + augmentation
```

---

## Evaluation

All metrics (mIoU, Pixel Accuracy, Mean Dice, FWIoU, Boundary IoU) are implemented in `evaluation_matrix/` and share a unified interface:

```bash
python evaluation_matrix/evaluate.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --ignore_index 255
```

---

## Stage 2: VLM Reasoning

The `VLM_generation/` module feeds segmentation masks from the best Stage 1 model (UNet++ EfficientNet-B4 V3) into GPT-4o for structured scene description.

```bash
cd VLM_generation
python main.py --mode segmentation_guided   # RGB + mask
python main.py --mode rgb_only              # baseline
python evaluate_vlm_descriptions.py        # compare outputs
```

Key finding: segmentation guidance improves scene-level spatial grounding (layout, region delineation, inter-object relationships) but can reduce fine-grained instance-level detail for small or closely-spaced objects.

---

## Results Summary

| Model                        | mIoU  | Pixel Acc. | Mean Dice |
|------------------------------|-------|------------|-----------|
| UNet (Boundary loss)         | 0.395 | 0.879      | —         |
| DeepLabV3+ V3 (R-101 + TTA)  | 0.493 | 0.914      | —         |
| SegFormer-B1 V3 (+ TTA)      | 0.496 | 0.897      | 0.606     |
| **UNet++ EfficientNet-B4 V3**| **0.521** | 0.887  | 0.597     |
| YOLO26s-seg                  | 0.441 | 0.846      | 0.556     |

---

## Dependencies

```bash
pip install -r UNet_plusplus/requirements.txt
# or for SegFormer:
conda env create -f UNet_plusplus/environment.yml
```

Key dependencies: `torch`, `segmentation-models-pytorch`, `transformers`, `albumentations`, `timm`, `ultralytics`.

---

## Key Findings

- **Training recipe matters as much as architecture**: the V1→V3 progressive ladder delivers consistent gains across all model families by addressing class imbalance, augmentation, and optimization independently.
- **Transfer learning is critical**: pretrained encoders outperform scratch models by 0.08–0.10 mIoU on the small CamVid training set.
- **Segmentation as VLM prior**: semantic masks improve scene-level spatial reasoning but introduce a granularity ceiling — category-level masks cannot separate instances, limiting fine-grained counting.

---

## License

See [LICENSE](LICENSE) for details.