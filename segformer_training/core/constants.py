from __future__ import annotations

from typing import Dict, List, Set, Tuple

CAMVID_CLASSES: List[str] = [
    "sky",
    "building",
    "pole",
    "road",
    "pavement",
    "tree",
    "signsymbol",
    "fence",
    "car",
    "pedestrian",
    "bicyclist",
]

NUM_CLASSES: int = len(CAMVID_CLASSES)
IGNORE_INDEX: int = 255

ID2LABEL: Dict[int, str] = {index: name for index, name in enumerate(CAMVID_CLASSES)}
LABEL2ID: Dict[str, int] = {name: index for index, name in ID2LABEL.items()}

CAMVID_COLORMAP: List[Tuple[int, int, int]] = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
]

BACKBONE_CHANNELS: Dict[str, List[int]] = {
    "nvidia/mit-b0": [32, 64, 160, 256],
    "nvidia/mit-b1": [64, 128, 320, 512],
    "nvidia/mit-b2": [64, 128, 320, 512],
    "nvidia/mit-b3": [64, 128, 320, 512],
    "nvidia/mit-b5": [64, 128, 320, 512],
}

SEGFORMER_BACKBONE_DEPTHS: Dict[str, List[int]] = {
    "nvidia/mit-b0": [2, 2, 2, 2],
    "nvidia/mit-b1": [2, 2, 2, 2],
    "nvidia/mit-b2": [3, 4, 6, 3],
    "nvidia/mit-b3": [3, 4, 18, 3],
    "nvidia/mit-b5": [3, 6, 40, 3],
}

# HuggingFace's default decoder_hidden_size per SegFormer backbone.
# B0: 256 (paper spec) — lightweight student
# B1-B5: 768 (paper spec) — larger decoder for bigger variants
# Used when building teacher model from a saved full_model.pt checkpoint.
DECODER_HIDDEN_SIZES: Dict[str, int] = {
    "nvidia/mit-b0": 256,
    "nvidia/mit-b1": 256,
    "nvidia/mit-b2": 768,
    "nvidia/mit-b3": 768,
    "nvidia/mit-b5": 768,
}

PHASE1_ALLOWED_REPORT_TO = {"none", "tensorboard", "wandb"}
ALLOWED_REPORT_TO = {"none", "tensorboard", "wandb"}
ALLOWED_AUGMENTATION_STRATEGIES = {"basic", "strong", "classmix"}
ALLOWED_LOSS_TYPES = {"ce", "dice", "focal", "combined"}
ALLOWED_KD_METHODS = {"logit", "decoder", "multilevel"}
EXPERIMENT_STATUSES = {"initialized", "training", "trained", "testing", "reporting", "completed", "crashed"}
RESUME_MUTABLE_FIELDS: Set[str] = {
    "training.num_train_epochs",
    "training.logging_steps",
    "training.report_to",
    "training.save_total_limit",
    "evaluation.tta.enabled",
    "evaluation.tta.scales",
    "evaluation.tta.flip",
    "evaluation.save_predictions",
    "evaluation.num_visualizations",
    "evaluation.compute_confusion_matrix",
}

TRAINING_HISTORY_HEADERS: List[str] = [
    "timestamp",
    "step",
    "epoch",
    "event",
    "train_loss",
    "eval_loss",
    "eval_mean_iou",
    "eval_overall_accuracy",
    "lr",
    "best_so_far",
    "checkpoint_path",
]

# --- Phase tracking ---

ACTIVE_PHASES: Set[str] = {"training", "validating", "testing", "reporting"}
ACTIVE_INDEX_STATUSES: Set[str] = {"initialized", "training", "trained", "testing", "reporting"}

ALL_PHASES: List[str] = [
    "initialized", "training", "validating", "testing", "reporting", "completed", "crashed"
]

STALE_THRESHOLDS_SECONDS: Dict[str, int] = {
    "training": 60,
    "validating": 120,
    "testing": 180,
    "reporting": 180,
}
