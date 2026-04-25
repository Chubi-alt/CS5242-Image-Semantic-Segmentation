# SegFormer Training Code Package

This directory is the **minimal training code package** for the CS5242 Group 20 SegFormer-B1/B2 fine-tuning ladder experiments.
It contains only the core code and configuration needed to run **training + model evaluation**. It does **not** include report generation, cross-run comparison, slides, or any analysis code; it also does **not** include test code or raw data.

---

## Directory Layout

```
segformer_training/
├── README.md                 this document
├── requirements.txt          pinned dependency versions
├── configs/                  all experiment configs (YAML)
│   ├── base.yaml             shared base (inherited by every exp_*.yaml)
│   ├── baseline.yaml         plain CE baseline
│   ├── exp_aug_*.yaml        V1: data-level (ClassMix / strong augmentation)
│   ├── exp_loss_*.yaml       V2: loss-level (Focal / Dice / combined / OHEM)
│   ├── exp_all*.yaml         V3: full ladder (with TTA)
│   ├── exp_auxiliary.yaml    auxiliary head experiment
│   ├── exp_b2_*.yaml         B2 experiments (including KD teacher pre-training)
│   └── exp_kd_*.yaml         KD distillation experiments (logit / decoder / multilevel / smoke)
├── core/                     core training modules
│   ├── __init__.py
│   ├── constants.py          classes / ignore_index / colormap / backbone channels
│   ├── augmentations.py      ClassMix + train/val transforms
│   ├── dataset.py            CamVid dataset + class_frequencies
│   ├── losses.py             CE / Dice / Focal / Lovász / boundary / OHEM / combined
│   ├── metrics.py            mIoU / per-class IoU / confusion matrix
│   ├── model.py              SegFormer B0/B1/B2 + aux head + inference bundle
│   ├── distillation.py       KD (logit / decoder / multilevel) — optional
│   ├── trainer.py            SegFormerTrainer (with KD forward hook)
│   ├── tta.py                multi-scale + HFlip test-time augmentation
│   ├── callbacks.py          StatusCallback / LocalMetricsCallback / CheckpointMarker
│   ├── config.py             YAML merging + validation + run-context resolution
│   ├── index.py              experiment_index.json read/write
│   └── utils.py              seed / status / checkpoint / atomic writes etc.
└── scripts/
    ├── __init__.py
    └── run.py                the only training entry point (reporting phase stripped)
```

---

## Environment Setup

```bash
cd segformer_training
pip install -r requirements.txt
```

Main dependencies: `torch==2.3.1` / `transformers==4.44.2` / `albumentations==1.4.3` / `segmentation-models-pytorch==0.3.4`.

---

## Preparing the Data

By default `configs/base.yaml` points to `data/camvid/` under the project root. If your data lives elsewhere, choose one of the two options below.

**Option A: edit a config (recommended)**
```bash
cp configs/local.yaml.example configs/local.yaml
# Edit configs/local.yaml and set data.data_dir: /abs/path/to/camvid
```

**Option B: override from the command line**
```bash
python scripts/run.py --config configs/baseline.yaml \
  --set data.data_dir=/abs/path/to/camvid
```

The CamVid data directory should contain three subdirectories `train/ val/ test/`, each with `images/` and `labels/` inside.

---

## Running Training

**Baseline (plain CE):**
```bash
python scripts/run.py --config configs/baseline.yaml
```

**V1 — data-level (ClassMix):**
```bash
python scripts/run.py --config configs/exp_aug_classmix.yaml
```

**V2 — loss-level (Focal):**
```bash
python scripts/run.py --config configs/exp_loss_focal.yaml
```

**V3 — full ladder + TTA:**
```bash
python scripts/run.py --config configs/exp_all_tta.yaml
```

**B2 baseline:**
```bash
python scripts/run.py --config configs/exp_b2_baseline.yaml
```

**KD (requires an existing B2 teacher checkpoint):**
```bash
python scripts/run.py --config configs/exp_kd_logit.yaml \
  --set kd.teacher_checkpoint=/path/to/b2_best/segformer
```

**Override arbitrary fields:**
```bash
python scripts/run.py --config configs/baseline.yaml \
  --set training.num_train_epochs=50 \
  --set training.learning_rate=1e-4 \
  --set experiment.name=my_run
```

**Resume training:**
```bash
python scripts/run.py --config configs/baseline.yaml --resume
# Or resume a specific run-id:
python scripts/run.py --config configs/baseline.yaml --resume --run-id 20260425_120000_baseline
```

---

## Training Artifacts

Each run produces the following under `results/<run_id>/`:
- `config_snapshot.yaml` — full config snapshot for this run
- `training_history.csv` — per-epoch loss / lr / val-mIoU
- `status.json` — live training status (phase / step / ETA / current metrics)
- `metrics.json` — final validation / test metrics
- `checkpoints/best/` — best checkpoint (contains a HuggingFace-style `segformer/` subdirectory)
- `checkpoints/last/` — most recent checkpoint

`results/experiment_index.json` aggregates metadata for every historical run.

---

## Differences From the Full Project

Compared with the original repository, this package **removes**:
- `core/reporting.py` / `core/result_analysis.py` — report generation and cross-run comparison
- `core/status_reader.py` / `core/model_loader.py` / `core/dataset_prep.py` — external CLI helpers
- All scripts under `scripts/` other than `run.py` (report / analyze / diagnose / inference / status)
- `tests/` — the test code
- The reporting phase at the tail of training that depends on `core/reporting` (already removed from `run.py`)

The training flow remains complete: **init → training → testing → completed** (the original `reporting` stage is skipped).
