# SegFormer 训练代码包（upload/）

本目录是 CS5242 Group 20 SegFormer-B1/B2 微调梯子实验的**最小训练代码包**。
仅包含跑通「训练 + 模型评估」所需的核心代码与配置；**不含**报告生成 / 跨 run 对比 / slides 等分析类代码，也**不含**测试代码与原始数据。

---

## 目录结构

```
upload/
├── README.md                 本说明
├── requirements.txt          依赖版本
├── configs/                  所有实验配置（YAML）
│   ├── base.yaml             共享 base（被其它 exp_*.yaml 继承）
│   ├── baseline.yaml         纯 CE baseline
│   ├── exp_aug_*.yaml        V1: data-level（ClassMix / 强增广）
│   ├── exp_loss_*.yaml       V2: loss-level（Focal / Dice / combined / OHEM）
│   ├── exp_all*.yaml         V3: 完整梯子（含 TTA）
│   ├── exp_auxiliary.yaml    aux head 实验
│   ├── exp_b2_*.yaml         B2 实验（含 KD teacher 预训练）
│   └── exp_kd_*.yaml         KD 蒸馏实验（logit / decoder / multilevel / smoke）
├── core/                     训练核心模块
│   ├── __init__.py
│   ├── constants.py          类别 / ignore_index / colormap / backbone channels
│   ├── augmentations.py      ClassMix + train/val transforms
│   ├── dataset.py            CamVid 数据集 + class_frequencies
│   ├── losses.py             CE / Dice / Focal / Lovász / boundary / OHEM / combined
│   ├── metrics.py            mIoU / per-class IoU / confusion matrix
│   ├── model.py              SegFormer B0/B1/B2 + aux head + inference bundle
│   ├── distillation.py       KD (logit / decoder / multilevel) — 可选
│   ├── trainer.py            SegFormerTrainer（含 KD forward hook）
│   ├── tta.py                多尺度 + HFlip 测试时增强
│   ├── callbacks.py          StatusCallback / LocalMetricsCallback / CheckpointMarker
│   ├── config.py             YAML 合并 + 校验 + run context 解析
│   ├── index.py              experiment_index.json 读写
│   └── utils.py              seed / status / checkpoint / 原子写入等
└── scripts/
    ├── __init__.py
    └── run.py                唯一训练入口（已剥离 reporting 阶段）
```

---

## 环境安装

```bash
cd upload
pip install -r requirements.txt
```

主要依赖：`torch==2.3.1` / `transformers==4.44.2` / `albumentations==1.4.3` / `segmentation-models-pytorch==0.3.4`。

---

## 准备数据

默认 `configs/base.yaml` 指向项目根目录下的 `data/camvid/`。若数据在别处，任选一种方式：

**方式 A：改 config（推荐）**
```bash
cp configs/local.yaml.example configs/local.yaml
# 编辑 configs/local.yaml，设置 data.data_dir: /abs/path/to/camvid
```

**方式 B：命令行覆写**
```bash
python scripts/run.py --config configs/baseline.yaml \
  --set data.data_dir=/abs/path/to/camvid
```

CamVid 数据目录应包含 `train/ val/ test/` 三个子目录，每个子目录下有 `images/` 与 `labels/`。

---

## 运行训练

**baseline（纯 CE）**：
```bash
python scripts/run.py --config configs/baseline.yaml
```

**V1 — data-level（ClassMix）**：
```bash
python scripts/run.py --config configs/exp_aug_classmix.yaml
```

**V2 — loss-level（Focal）**：
```bash
python scripts/run.py --config configs/exp_loss_focal.yaml
```

**V3 — 完整梯子 + TTA**：
```bash
python scripts/run.py --config configs/exp_all_tta.yaml
```

**B2 baseline**：
```bash
python scripts/run.py --config configs/exp_b2_baseline.yaml
```

**KD（需先有 B2 teacher checkpoint）**：
```bash
python scripts/run.py --config configs/exp_kd_logit.yaml \
  --set kd.teacher_checkpoint=/path/to/b2_best/segformer
```

**覆写任意字段**：
```bash
python scripts/run.py --config configs/baseline.yaml \
  --set training.num_train_epochs=50 \
  --set training.learning_rate=1e-4 \
  --set experiment.name=my_run
```

**续训**：
```bash
python scripts/run.py --config configs/baseline.yaml --resume
# 或指定具体 run-id：
python scripts/run.py --config configs/baseline.yaml --resume --run-id 20260425_120000_baseline
```

---

## 训练产物

每次 run 会在 `results/<run_id>/` 下生成：
- `config_snapshot.yaml` — 本次 run 的完整配置快照
- `training_history.csv` — 每 epoch 的 loss / lr / val-mIoU
- `status.json` — 实时训练状态（phase / step / ETA / 当前指标）
- `metrics.json` — 最终 validation / test 指标
- `checkpoints/best/` — 最佳 checkpoint（含 HuggingFace-style `segformer/` 子目录）
- `checkpoints/last/` — 最新 checkpoint

`results/experiment_index.json` 汇总所有历史 run 的元数据。

---

## 与完整项目的差异

相比原始仓库，本包**移除了**：
- `core/reporting.py` / `core/result_analysis.py` — 报告与跨 run 对比分析
- `core/status_reader.py` / `core/model_loader.py` / `core/dataset_prep.py` — 外部 CLI 辅助
- `scripts/` 下除 `run.py` 外所有脚本（report / analyze / diagnose / inference / status）
- `tests/` — 测试代码
- `core/reporting` 相关的训练末段 reporting phase（run.py 中已删除）

训练流程仍是完整的：**init → training → testing → completed**（跳过原 `reporting` 阶段）。
