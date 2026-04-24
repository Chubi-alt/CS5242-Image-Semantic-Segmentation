from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import importlib
import json
import os
import random
import shutil
import tempfile
import warnings

import numpy as np
import torch
import yaml
from PIL import Image

from core.constants import CAMVID_COLORMAP, IGNORE_INDEX, RESUME_MUTABLE_FIELDS, TRAINING_HISTORY_HEADERS


def atomic_write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def atomic_write_yaml(data: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def atomic_write_csv(
    rows: Iterable[Dict[str, Any]],
    path: str | Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    headers = fieldnames or TRAINING_HISTORY_HEADERS
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            normalized = {key: _normalize_csv_value(row.get(key, "")) for key in headers}
            writer.writerow(normalized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_json(path: str | Path) -> Dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def resolve_report_to(config: Any) -> str:
    requested = config.training.report_to
    if requested == "none":
        return "none"
    if requested == "tensorboard":
        try:
            importlib.import_module("tensorboard")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("training.report_to=tensorboard requires the 'tensorboard' package.") from exc
        return "tensorboard"
    if requested == "wandb":
        return safe_wandb_init(config)
    raise RuntimeError(f"Unsupported report_to value: {requested}")


def safe_wandb_init(config: Any) -> str:
    del config
    try:
        import wandb  # type: ignore
    except Exception:
        warnings.warn("wandb is not installed; falling back to report_to='none'.")
        os.environ["WANDB_DISABLED"] = "true"
        return "none"

    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode == "disabled":
        warnings.warn("WANDB_MODE=disabled; falling back to report_to='none'.")
        os.environ["WANDB_DISABLED"] = "true"
        return "none"
    if mode in {"offline", "dryrun"}:
        return "wandb"

    anonymous_mode = os.environ.get("WANDB_ANONYMOUS", "").strip().lower()
    if anonymous_mode in {"allow", "must"}:
        return "wandb"

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        api = getattr(wandb, "api", None)
        api_key = getattr(api, "api_key", None)
    if api_key:
        return "wandb"

    warnings.warn("wandb requested but no API key or offline mode was detected; falling back to report_to='none'.")
    os.environ["WANDB_DISABLED"] = "true"
    return "none"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preflight_checks(config: Any, run_context: Optional[Any] = None, data_dir: Optional[str] = None) -> None:
    errors: List[str] = []

    if run_context is not None:
        run_dir = Path(run_context.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _probe_writable_directory(run_dir, errors)
        _check_free_disk_space(run_dir, errors)

    if config.training.bf16 and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; forcing training.bf16=False.")
        config.training.bf16 = False

    root = Path(data_dir) if data_dir is not None else Path(config.data.dir)
    for split in ("train", "val"):
        _validate_split(root, split, errors)
    test_dir = root / "test"
    test_annot_dir = root / "testannot"
    if test_dir.exists() or test_annot_dir.exists():
        _validate_split(root, "test", errors)

    if run_context is not None and getattr(run_context, "resumed", False):
        if config.model.resume_from is not None:
            errors.append("--resume and model.resume_from cannot be used together.")
        checkpoint = getattr(run_context, "resume_from_checkpoint", None)
        if not checkpoint:
            errors.append("Resume requested but no valid checkpoint was resolved.")
        else:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists() or not (checkpoint_path / ".save_complete").exists():
                errors.append(f"Resume checkpoint is invalid: {checkpoint_path}")
        snapshot_path = Path(run_context.run_dir) / "config_snapshot.yaml"
        if not snapshot_path.exists():
            errors.append(f"Missing config snapshot for resume target: {snapshot_path}")
        else:
            from core.config import config_to_dict, load_config_snapshot

            previous_config = load_config_snapshot(snapshot_path, validate=False)
            errors.extend(compare_resume_config_compatibility(config_to_dict(previous_config), config_to_dict(config)))
            if config.training.num_train_epochs < previous_config.training.num_train_epochs:
                errors.append("training.num_train_epochs cannot decrease during --resume.")

    if config.model.resume_from is not None:
        errors.extend(validate_resume_export(config.model.resume_from, expected_backbone=config.model.backbone))

    if errors:
        raise RuntimeError("\n".join(errors))


def compare_resume_config_compatibility(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    previous_flat = flatten_mapping(previous)
    current_flat = flatten_mapping(current)
    errors: List[str] = []
    for key in sorted(set(previous_flat) | set(current_flat)):
        if _is_mutable_resume_field(key):
            continue
        if previous_flat.get(key) != current_flat.get(key):
            errors.append(f"--resume incompatible config change at '{key}': {previous_flat.get(key)} -> {current_flat.get(key)}")
    return errors


def validate_resume_export(resume_from: str | Path, expected_backbone: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        return [f"model.resume_from path does not exist: {path}"]
    if path.is_file():
        if path.name != "full_model.pt" or path.parent.name != "best":
            return ["model.resume_from must point to best/full_model.pt or best/segformer/."]
    elif path.is_dir():
        if path.name != "segformer" or path.parent.name != "best":
            return ["model.resume_from must point to best/full_model.pt or best/segformer/."]
        if not has_processor_artifacts(path):
            snapshot_path = find_snapshot_for_export_path(path)
            if snapshot_path is None:
                errors.append(f"segformer export is missing processor artifacts and no config_snapshot.yaml was found: {path}")
            else:
                warnings.warn(f"segformer export missing processor artifacts, falling back to config_snapshot.yaml: {path}")
    else:
        return [f"Unsupported model.resume_from path: {path}"]

    snapshot_path = find_snapshot_for_export_path(path)
    if expected_backbone is not None and snapshot_path is not None:
        from core.config import load_config_snapshot

        source_config = load_config_snapshot(snapshot_path, validate=False)
        if source_config.model.backbone != expected_backbone:
            errors.append(
                f"Backbone mismatch between current config ({expected_backbone}) and resume source ({source_config.model.backbone})."
            )
    return errors


def find_valid_checkpoint(checkpoint_dir: str | Path) -> Optional[str]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None
    candidates = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-") and path.name.split("-", 1)[1].isdigit()
        ],
        key=lambda item: int(item.name.split("-", 1)[1]),
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / ".save_complete").exists():
            return str(candidate)
    return None


def has_processor_artifacts(segformer_dir: str | Path) -> bool:
    directory = Path(segformer_dir)
    return (directory / "preprocessor_config.json").exists()


def find_snapshot_for_export_path(export_path: str | Path) -> Optional[Path]:
    path = Path(export_path).expanduser().resolve()
    try:
        run_dir = path.parents[2]
    except IndexError:
        return None
    snapshot_path = run_dir / "config_snapshot.yaml"
    return snapshot_path if snapshot_path.exists() else None


def parse_run_id_from_export_path(export_path: str | Path) -> Optional[str]:
    path = Path(export_path).expanduser().resolve()
    try:
        return path.parents[2].name
    except IndexError:
        return None


def flatten_mapping(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def mask_to_color_image(mask: np.ndarray) -> Image.Image:
    height, width = mask.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in enumerate(CAMVID_COLORMAP):
        canvas[mask == class_id] = color
    canvas[mask == IGNORE_INDEX] = (0, 0, 0)
    return Image.fromarray(canvas)


def save_prediction_triptych(
    output_path: str | Path,
    image: Image.Image,
    label: np.ndarray,
    prediction: np.ndarray,
) -> None:
    source = image.convert("RGB")
    gt_rgb = mask_to_color_image(label)
    pred_rgb = mask_to_color_image(prediction)
    width, height = source.size
    canvas = Image.new("RGB", (width * 3, height), color=(255, 255, 255))
    canvas.paste(source, (0, 0))
    canvas.paste(gt_rgb.resize((width, height), Image.NEAREST), (width, 0))
    canvas.paste(pred_rgb.resize((width, height), Image.NEAREST), (width * 2, 0))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def ensure_mpl_config_dir() -> str:
    current = os.environ.get("MPLCONFIGDIR")
    if current and Path(current).exists():
        return current
    target = Path(tempfile.gettempdir()) / "segexpress-mplconfig"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(target)
    return str(target)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value}"
    return str(value)


def _probe_writable_directory(run_dir: Path, errors: List[str]) -> None:
    probe = run_dir / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        errors.append(f"Output directory is not writable: {run_dir} ({exc})")


def _check_free_disk_space(run_dir: Path, errors: List[str]) -> None:
    free_bytes = shutil.disk_usage(run_dir).free
    if free_bytes < 2 * 1024 * 1024 * 1024:
        errors.append(f"At least 2GB free space is required under {run_dir}.")


def _validate_split(root: Path, split: str, errors: List[str]) -> None:
    image_dir = root / split
    label_dir = root / f"{split}annot"
    if not image_dir.exists():
        errors.append(f"Missing split directory: {image_dir}")
        return
    if not label_dir.exists():
        errors.append(f"Missing annotation directory: {label_dir}")
        return

    image_paths = sorted(path.name for path in image_dir.glob("*.png"))
    label_paths = sorted(path.name for path in label_dir.glob("*.png"))
    if not image_paths:
        errors.append(f"No .png files found in split: {image_dir}")
        return
    if image_paths != label_paths:
        missing_labels = sorted(set(image_paths) - set(label_paths))
        missing_images = sorted(set(label_paths) - set(image_paths))
        errors.append(
            f"Image/label mismatch in split '{split}'. "
            f"Missing labels: {missing_labels[:3]} Missing images: {missing_images[:3]}"
        )
        return

    allowed_values = set(range(11)) | {11, IGNORE_INDEX}
    sample_paths = list(label_dir.glob("*.png"))[:10]
    for sample_path in sample_paths:
        label = np.asarray(Image.open(sample_path))
        invalid = set(np.unique(label).tolist()) - allowed_values
        if invalid:
            errors.append(f"Invalid label ids in {sample_path}: {sorted(invalid)}")
            break


def _is_mutable_resume_field(field_path: str) -> bool:
    for allowed_prefix in RESUME_MUTABLE_FIELDS:
        if field_path == allowed_prefix or field_path.startswith(f"{allowed_prefix}."):
            return True
    return False


def write_status_json(
    run_dir: "str | Path",
    run_id: str,
    experiment_name: str,
    phase: str,
    phase_step: Optional[int],
    phase_total_steps: Optional[int],
    phase_message: str,
    current_epoch: int,
    total_epochs: int,
    global_step: int,
    total_steps: int,
    steps_per_epoch: int,
    steps_per_epoch_source: str,
    train_loss: Optional[float],
    last_eval_miou: Optional[float],
    best_eval_miou: Optional[float],
    eta_seconds: Optional[float],
    eta_source: Optional[str],
    hostname: str,
    pid: int,
    device: str,
) -> None:
    """Write status.json atomically. Silently ignores write failures."""
    phase_progress = (
        round(phase_step / phase_total_steps, 4)
        if phase_step is not None and phase_total_steps and phase_total_steps > 0
        else None
    )
    data = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "run_dir": str(Path(run_dir).resolve()),
        "hostname": hostname,
        "pid": pid,
        "device": device,
        "phase": phase,
        "phase_step": phase_step,
        "phase_total_steps": phase_total_steps,
        "phase_progress": phase_progress,
        "phase_message": phase_message,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "global_step": global_step,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "steps_per_epoch_source": steps_per_epoch_source,
        "train_loss": train_loss,
        "last_eval_miou": last_eval_miou,
        "best_eval_miou": best_eval_miou,
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "eta_source": eta_source,
        "last_heartbeat": utc_timestamp(),
    }
    try:
        atomic_write_json(data, Path(run_dir) / "status.json")
    except Exception as exc:
        import sys
        print(f"[status] write failed (non-fatal): {exc}", file=sys.stderr)
