from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints
import re

import yaml

from core.constants import (
    ALLOWED_AUGMENTATION_STRATEGIES,
    ALLOWED_KD_METHODS,
    ALLOWED_LOSS_TYPES,
    ALLOWED_REPORT_TO,
    BACKBONE_CHANNELS,
)


T = TypeVar("T")


class ConfigValidationError(ValueError):
    """Raised when configuration is invalid."""


@dataclass
class AuxiliaryHeadConfig:
    enabled: bool = False
    stages: List[int] = field(default_factory=lambda: [2, 3])
    weight: float = 0.4
    channels: int = 256


@dataclass
class KDConfig:
    enabled: bool = False
    method: str = "logit"              # "logit" | "decoder" | "multilevel"
    teacher_checkpoint: Optional[str] = None
    temperature: float = 4.0
    alpha: float = 0.5                 # weight of KD loss term added to task loss
    feature_weight: float = 0.5        # decoder feature MSE weight (Methods 2 & 3)
    stage_weight: float = 0.3          # backbone stage MSE weight (Method 3 only)


@dataclass
class ModelConfig:
    backbone: str = "nvidia/mit-b0"
    pretrained: bool = True
    num_labels: int = 11
    resume_from: Optional[str] = None
    auxiliary_heads: AuxiliaryHeadConfig = field(default_factory=AuxiliaryHeadConfig)


@dataclass
class BoundaryConfig:
    enabled: bool = False
    weight: float = 0.2
    sigma: float = 5.0


@dataclass
class OHEMConfig:
    enabled: bool = False
    ratio: float = 0.7


@dataclass
class LossComponentConfig:
    type: str = "ce"
    weight: float = 1.0


@dataclass
class LossConfig:
    type: str = "ce"
    components: List[LossComponentConfig] = field(default_factory=list)
    focal_gamma: float = 2.0
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    ohem: OHEMConfig = field(default_factory=OHEMConfig)
    class_weights: bool = False


@dataclass
class RandomScaleConfig:
    enabled: bool = False
    range: List[float] = field(default_factory=lambda: [0.5, 2.0])


@dataclass
class ColorJitterConfig:
    enabled: bool = False
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.1


@dataclass
class GaussianBlurConfig:
    enabled: bool = False
    kernel_range: List[int] = field(default_factory=lambda: [3, 7])


@dataclass
class ClassMixConfig:
    prob: float = 0.5
    num_classes: Optional[int] = None


@dataclass
class AugmentationConfig:
    strategy: str = "basic"
    image_size: int = 512
    horizontal_flip: bool = True
    random_scale: RandomScaleConfig = field(default_factory=RandomScaleConfig)
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    gaussian_blur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)
    classmix: ClassMixConfig = field(default_factory=ClassMixConfig)


@dataclass
class TTAConfig:
    enabled: bool = False
    scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25, 1.5])
    flip: bool = True


@dataclass
class EvaluationConfig:
    tta: TTAConfig = field(default_factory=TTAConfig)
    save_predictions: bool = True
    num_visualizations: int = -1
    compute_confusion_matrix: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-5
    num_train_epochs: int = 50
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    bf16: bool = True
    fp16: bool = False
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_mean_iou"
    greater_is_better: bool = True
    dataloader_num_workers: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    report_to: str = "wandb"


@dataclass
class DataConfig:
    dir: str = "./data/camvid"
    ignore_index: int = 255

    def resolve_data_dir(self, config_path: "str | Path") -> Path:
        base = Path(config_path).resolve().parent
        p = Path(self.dir)
        return (base / p).resolve() if not p.is_absolute() else p.resolve()


@dataclass
class PathsConfig:
    results_dir: str = "../results"
    checkpoints_dir: Optional[str] = None

    def resolve_results_dir(self, config_path: "str | Path") -> Path:
        base = Path(config_path).resolve().parent
        p = Path(self.results_dir)
        return (base / p).resolve() if not p.is_absolute() else p.resolve()

    def resolve_checkpoints_dir(self, config_path: "str | Path") -> Optional[Path]:
        if self.checkpoints_dir is None:
            return None
        base = Path(config_path).resolve().parent
        p = Path(self.checkpoints_dir)
        return (base / p).resolve() if not p.is_absolute() else p.resolve()


@dataclass
class ExperimentMeta:
    name: str = "default"
    seed: int = 42
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    kd: KDConfig = field(default_factory=KDConfig)


@dataclass
class RunContext:
    run_id: str
    run_dir: str
    checkpoint_dir: str
    resume_from_checkpoint: Optional[str] = None
    resumed: bool = False
    continued_from: Optional[str] = None


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    data = _load_yaml_with_bases(path)
    for override in overrides or []:
        _apply_override(data, override)
    config = _config_from_mapping(data)
    validate_config(config)
    return config


def load_config_snapshot(snapshot_path: str | Path, validate: bool = True) -> ExperimentConfig:
    path = Path(snapshot_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config snapshot not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Config root must be a mapping: {path}")
    config = _config_from_mapping(raw)
    if validate:
        validate_config(config)
    return config


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)


def generate_run_id(config: ExperimentConfig) -> str:
    from datetime import datetime
    import secrets

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", config.experiment.name).strip("_") or "run"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{safe_name}_{timestamp}_{suffix}"


def _resolve_checkpoint_dir_for_resume(
    run_dir: Path,
    cli_checkpoints_dir: Optional[str],
    run_id: str,
) -> Path:
    """Three-tier priority: CLI arg > run_meta.json > fallback."""
    import json
    # Tier 1: explicit CLI/config
    if cli_checkpoints_dir is not None:
        return Path(cli_checkpoints_dir) / run_id

    # Tier 2: run_meta.json
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stored = meta.get("checkpoint_dir")
            if stored and Path(stored).exists():
                return Path(stored)
        except Exception:
            pass

    # Tier 3: fallback to legacy layout
    return run_dir / "checkpoints"


def resolve_run_context(
    config: ExperimentConfig,
    resume: bool = False,
    run_id: Optional[str] = None,
    results_dir: str = "results",
    checkpoints_dir: Optional[str] = None,
) -> RunContext:
    import socket
    from core.utils import atomic_write_json, utc_timestamp

    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    if not resume:
        if run_id is not None:
            raise ValueError("--run-id can only be used together with --resume.")
        resolved_run_id = generate_run_id(config)
        run_dir = results_root / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint dir: external root/{run_id} OR run_dir/checkpoints
        if checkpoints_dir is not None:
            ckpt_dir = Path(checkpoints_dir) / resolved_run_id
        else:
            ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Write run_meta.json (atomic)
        run_meta = {
            "run_id": resolved_run_id,
            "run_dir": str(run_dir.resolve()),
            "checkpoint_dir": str(ckpt_dir.resolve()),
            "checkpoint_root": checkpoints_dir or str((run_dir / "checkpoints").resolve()),
            "checkpoint_layout_version": 1,
            "config_snapshot_path": str((run_dir / "config_snapshot.yaml").resolve()),
            "created_hostname": socket.gethostname(),
            "created_at": utc_timestamp(),
        }
        atomic_write_json(run_meta, run_dir / "run_meta.json")

        return RunContext(
            run_id=resolved_run_id,
            run_dir=str(run_dir),
            checkpoint_dir=str(ckpt_dir),
            resumed=False,
        )

    # --- resume path ---
    resolved_run_id = run_id or _find_latest_run_id_for_experiment(config.experiment.name, results_root)
    if resolved_run_id is None:
        raise FileNotFoundError(f"No existing run found for experiment '{config.experiment.name}'.")
    run_dir = results_root / resolved_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Three-tier checkpoint resolution
    ckpt_dir = _resolve_checkpoint_dir_for_resume(run_dir, checkpoints_dir, resolved_run_id)

    from core.utils import find_valid_checkpoint
    resume_from_checkpoint = find_valid_checkpoint(ckpt_dir)
    if resume_from_checkpoint is None:
        raise FileNotFoundError(f"No valid checkpoint found under: {ckpt_dir}")
    return RunContext(
        run_id=resolved_run_id,
        run_dir=str(run_dir),
        checkpoint_dir=str(ckpt_dir),
        resume_from_checkpoint=resume_from_checkpoint,
        resumed=True,
    )


def save_config_snapshot(config: ExperimentConfig, output_dir: str) -> str:
    from core.utils import atomic_write_yaml

    output_path = Path(output_dir) / "config_snapshot.yaml"
    atomic_write_yaml(config_to_dict(config), output_path)
    return str(output_path)


def validate_config(config: ExperimentConfig) -> None:
    errors: List[str] = []

    if config.model.backbone not in BACKBONE_CHANNELS:
        errors.append(f"Unsupported backbone: {config.model.backbone}")
    if config.model.num_labels != 11:
        errors.append("CamVid expects model.num_labels=11.")
    if any(stage < 0 or stage > 3 for stage in config.model.auxiliary_heads.stages):
        errors.append("model.auxiliary_heads.stages must stay within [0, 3].")
    if config.model.auxiliary_heads.channels <= 0:
        errors.append("model.auxiliary_heads.channels must be positive.")
    if config.model.auxiliary_heads.weight < 0:
        errors.append("model.auxiliary_heads.weight must be non-negative.")
    if config.model.resume_from is not None:
        resume_path = Path(config.model.resume_from).expanduser()
        if not resume_path.exists():
            errors.append(f"model.resume_from path does not exist: {resume_path}")
        elif not _is_supported_resume_export_path(resume_path):
            errors.append("model.resume_from must point to best/full_model.pt or best/segformer/.")

    if config.loss.type not in ALLOWED_LOSS_TYPES:
        allowed_loss = ", ".join(sorted(ALLOWED_LOSS_TYPES))
        errors.append(f"loss.type must be one of: {allowed_loss}")
    if config.loss.focal_gamma <= 0:
        errors.append("loss.focal_gamma must be positive.")
    if config.loss.boundary.weight < 0:
        errors.append("loss.boundary.weight must be non-negative.")
    if config.loss.boundary.sigma <= 0:
        errors.append("loss.boundary.sigma must be positive.")
    if config.loss.ohem.enabled and not (0 < config.loss.ohem.ratio <= 1):
        errors.append("loss.ohem.ratio must be within (0, 1].")
    if config.loss.type == "combined":
        if not config.loss.components:
            errors.append("combined loss requires non-empty loss.components.")
        component_weight_sum = sum(component.weight for component in config.loss.components)
        if component_weight_sum <= 0:
            errors.append("combined loss requires component weights to sum to > 0.")
        for component in config.loss.components:
            if component.type not in {"ce", "dice", "focal"}:
                errors.append(f"Unsupported combined loss component: {component.type}")
            if component.weight <= 0:
                errors.append("combined loss component weights must be positive.")
    elif config.loss.components:
        errors.append("loss.components is only valid when loss.type='combined'.")
    if config.loss.ohem.enabled and not _config_has_ce_component(config.loss):
        errors.append("OHEM requires a CE loss term.")

    if config.augmentation.strategy not in ALLOWED_AUGMENTATION_STRATEGIES:
        allowed_aug = ", ".join(sorted(ALLOWED_AUGMENTATION_STRATEGIES))
        errors.append(f"augmentation.strategy must be one of: {allowed_aug}")
    if config.augmentation.image_size % 32 != 0:
        errors.append("augmentation.image_size must be divisible by 32.")
    if not (0.0 <= config.augmentation.classmix.prob <= 1.0):
        errors.append("augmentation.classmix.prob must stay within [0, 1].")
    if config.augmentation.classmix.num_classes is not None and config.augmentation.classmix.num_classes <= 0:
        errors.append("augmentation.classmix.num_classes must be positive when set.")
    if len(config.augmentation.random_scale.range) != 2 or any(value <= 0 for value in config.augmentation.random_scale.range):
        errors.append("augmentation.random_scale.range must contain two positive values.")
    if len(config.augmentation.gaussian_blur.kernel_range) != 2 or any(value <= 0 for value in config.augmentation.gaussian_blur.kernel_range):
        errors.append("augmentation.gaussian_blur.kernel_range must contain two positive integers.")

    if config.evaluation.num_visualizations < -1:
        errors.append("evaluation.num_visualizations must be -1, 0, or a positive integer.")
    if not config.evaluation.tta.scales or any(scale <= 0 for scale in config.evaluation.tta.scales):
        errors.append("evaluation.tta.scales must be a non-empty list of positive values.")

    if config.training.report_to not in ALLOWED_REPORT_TO:
        allowed = ", ".join(sorted(ALLOWED_REPORT_TO))
        errors.append(f"training.report_to must be one of: {allowed}")
    if config.training.eval_strategy not in {"steps", "epoch"}:
        errors.append("training.eval_strategy must be 'steps' or 'epoch'.")
    if config.training.save_strategy not in {"steps", "epoch"}:
        errors.append("training.save_strategy must be 'steps' or 'epoch'.")
    if config.training.eval_steps <= 0:
        errors.append("training.eval_steps must be positive.")
    if config.training.save_steps <= 0:
        errors.append("training.save_steps must be positive.")
    if config.training.logging_steps <= 0:
        errors.append("training.logging_steps must be positive.")
    if config.training.num_train_epochs <= 0:
        errors.append("training.num_train_epochs must be positive.")
    if config.training.per_device_train_batch_size <= 0 or config.training.per_device_eval_batch_size <= 0:
        errors.append("training batch sizes must be positive.")
    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be positive.")
    if config.training.warmup_ratio < 0:
        errors.append("training.warmup_ratio must be non-negative.")
    if config.training.weight_decay < 0:
        errors.append("training.weight_decay must be non-negative.")

    if config.data.ignore_index != 255:
        errors.append("CamVid expects data.ignore_index=255.")
    if not config.data.dir:
        errors.append("data.dir must not be empty.")

    if config.kd.enabled:
        if config.kd.method not in ALLOWED_KD_METHODS:
            allowed_kd = ", ".join(sorted(ALLOWED_KD_METHODS))
            errors.append(f"kd.method must be one of: {allowed_kd}")
        if config.kd.teacher_checkpoint is None:
            errors.append("kd.teacher_checkpoint must be set when kd.enabled=True.")
        else:
            teacher_path = Path(config.kd.teacher_checkpoint).expanduser()
            if not teacher_path.exists():
                errors.append(
                    f"kd.teacher_checkpoint does not exist: {teacher_path}"
                )
            elif not _is_supported_resume_export_path(teacher_path):
                errors.append(
                    "kd.teacher_checkpoint must point to best/full_model.pt "
                    f"(got: {teacher_path})"
                )
        if config.kd.temperature <= 0:
            errors.append("kd.temperature must be positive.")
        if not (0 < config.kd.alpha < 1):
            errors.append("kd.alpha must be in (0, 1).")
        if config.kd.feature_weight < 0:
            errors.append("kd.feature_weight must be non-negative.")
        if config.kd.stage_weight < 0:
            errors.append("kd.stage_weight must be non-negative.")
        if config.model.auxiliary_heads.enabled:
            errors.append(
                "kd.enabled=True is incompatible with model.auxiliary_heads.enabled=True. "
                "Student must be plain B0."
            )

    if errors:
        raise ConfigValidationError("\n".join(errors))


def _find_latest_run_id_for_experiment(experiment_name: str, results_root: Path) -> Optional[str]:
    from core.index import load_experiment_index

    try:
        index = load_experiment_index(results_dir=str(results_root))
    except Exception:
        index = []
    matching = [entry for entry in index if entry.experiment_name == experiment_name]
    if matching:
        matching.sort(key=lambda item: (item.started_at or "", item.run_id))
        return matching[-1].run_id

    candidates: List[tuple[str, str]] = []
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue
        snapshot_path = run_dir / "config_snapshot.yaml"
        if not snapshot_path.exists():
            continue
        try:
            snapshot = load_config_snapshot(snapshot_path, validate=False)
        except Exception:
            continue
        if snapshot.experiment.name == experiment_name:
            started_hint = snapshot_path.stat().st_mtime_ns
            candidates.append((str(started_hint), run_dir.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _load_yaml_with_bases(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Config root must be a mapping: {path}")

    base_ref = raw.pop("_base_", None)
    if base_ref in (None, "", "null"):
        return raw

    base_path = (path.parent / str(base_ref)).resolve()
    base_data = _load_yaml_with_bases(base_path)
    return deep_merge(base_data, raw)


def _apply_override(data: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigValidationError(f"Invalid override '{override}'. Expected key=value.")
    key_path, raw_value = override.split("=", 1)
    parsed_value = yaml.safe_load(raw_value)
    cursor: Dict[str, Any] = data
    parts = key_path.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ConfigValidationError(f"Override path '{key_path}' crosses non-mapping key '{part}'.")
        cursor = next_value
    cursor[parts[-1]] = parsed_value


def _config_from_mapping(data: Dict[str, Any]) -> ExperimentConfig:
    return _dataclass_from_dict(ExperimentConfig, data)


def _dataclass_from_dict(cls: Type[T], value: Any) -> T:
    if is_dataclass(cls):
        if value is None:
            return cls()  # type: ignore[misc]
        type_hints = get_type_hints(cls)
        kwargs = {}
        for field_def in fields(cls):
            raw = value.get(field_def.name) if isinstance(value, dict) else None
            annotation = type_hints.get(field_def.name, field_def.type)
            kwargs[field_def.name] = _convert_value(annotation, raw, field_def.default, field_def.default_factory)
        return cls(**kwargs)  # type: ignore[misc]
    raise TypeError(f"{cls} is not a dataclass type")


def _convert_value(annotation: Any, raw: Any, default: Any, default_factory: Any) -> Any:
    if raw is None:
        if default is not MISSING:
            return default
        if default_factory is not MISSING:
            return default_factory()
        return None

    origin = get_origin(annotation)
    if origin is Union:
        args = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
        if not args:
            return raw
        return _convert_value(args[0], raw, None, None)
    if origin in {list, List}:
        (item_type,) = get_args(annotation) or (Any,)
        return [_convert_value(item_type, item, None, None) for item in raw]
    if is_dataclass(annotation):
        return _dataclass_from_dict(annotation, raw)
    if annotation is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        raise ConfigValidationError(f"Expected a boolean value, got {raw!r}.")
    if annotation is int:
        if isinstance(raw, bool):
            raise ConfigValidationError(f"Expected an integer value, got {raw!r}.")
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f"Expected an integer value, got {raw!r}.") from exc
    if annotation is float:
        if isinstance(raw, bool):
            raise ConfigValidationError(f"Expected a float value, got {raw!r}.")
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f"Expected a float value, got {raw!r}.") from exc
    if annotation is str:
        return str(raw)
    return raw


def _config_has_ce_component(loss_config: LossConfig) -> bool:
    if loss_config.type == "ce":
        return True
    if loss_config.type == "combined":
        return any(component.type == "ce" for component in loss_config.components)
    return False


def _is_supported_resume_export_path(path: Path) -> bool:
    normalized = path.resolve()
    if normalized.is_file():
        return normalized.name == "full_model.pt" and normalized.parent.name == "best"
    if normalized.is_dir():
        return normalized.name == "segformer" and normalized.parent.name == "best"
    return False
