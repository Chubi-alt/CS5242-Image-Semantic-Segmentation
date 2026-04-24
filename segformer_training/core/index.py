from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
import json

from core.config import load_config_snapshot
from core.constants import EXPERIMENT_STATUSES
from core.utils import atomic_write_json


@dataclass
class ExperimentIndexEntry:
    run_id: str
    experiment_name: str
    status: str
    tags: List[str]
    config_path: str
    snapshot_path: str
    best_miou: Optional[float]
    best_checkpoint: Optional[str]
    started_at: str
    completed_at: Optional[str]
    continued_from: Optional[str]
    latest_requested_epochs: Optional[int]
    training_time_minutes: Optional[float]
    trainable_params: Optional[int]
    backbone: Optional[str] = None          # new, default None for backward compat


def load_experiment_index(results_dir: str = "results") -> List[ExperimentIndexEntry]:
    path = Path(results_dir) / "experiment_index.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in data:
        item.setdefault("backbone", None)   # backward compat
        entries.append(ExperimentIndexEntry(**item))
    return entries


def update_experiment_index(entry: ExperimentIndexEntry, results_dir: str = "results") -> None:
    if entry.status not in EXPERIMENT_STATUSES:
        raise ValueError(f"Unsupported experiment status: {entry.status}")
    entries = load_experiment_index(results_dir=results_dir)
    replaced = False
    for index, existing in enumerate(entries):
        if existing.run_id == entry.run_id:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    atomic_write_json([asdict(item) for item in entries], Path(results_dir) / "experiment_index.json")


def query_experiments(
    index: List[ExperimentIndexEntry],
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    min_miou: Optional[float] = None,
) -> List[ExperimentIndexEntry]:
    filtered = list(index)
    if tags:
        tag_set = set(tags)
        filtered = [entry for entry in filtered if tag_set.issubset(set(entry.tags))]
    if status is not None:
        filtered = [entry for entry in filtered if entry.status == status]
    if min_miou is not None:
        filtered = [entry for entry in filtered if entry.best_miou is not None and entry.best_miou >= min_miou]
    return filtered


def rebuild_experiment_index(results_dir: str = "results") -> List[ExperimentIndexEntry]:
    results_root = Path(results_dir)
    entries: List[ExperimentIndexEntry] = []
    if not results_root.exists():
        atomic_write_json([], results_root / "experiment_index.json")
        return entries

    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        snapshot_path = run_dir / "config_snapshot.yaml"
        if not snapshot_path.exists():
            continue
        try:
            config = load_config_snapshot(snapshot_path, validate=False)
        except Exception:
            continue

        metrics_path = run_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                metrics = {}

        status = metrics.get("status", "completed" if metrics else "initialized")
        entry = ExperimentIndexEntry(
            run_id=run_dir.name,
            experiment_name=config.experiment.name,
            status=status if status in EXPERIMENT_STATUSES else "initialized",
            tags=list(config.experiment.tags),
            config_path=str(snapshot_path),
            snapshot_path=str(snapshot_path),
            best_miou=_extract_best_miou(metrics),
            best_checkpoint=str(run_dir / "checkpoints" / "best") if (run_dir / "checkpoints" / "best").exists() else None,
            started_at=_isoformat_or_none(snapshot_path.stat().st_mtime),
            completed_at=_extract_completed_at(metrics),
            continued_from=metrics.get("continued_from"),
            latest_requested_epochs=config.training.num_train_epochs,
            training_time_minutes=_extract_training_minutes(metrics),
            trainable_params=_extract_trainable_params(metrics),
            backbone=config.model.backbone,
        )
        entries.append(entry)

    atomic_write_json([asdict(item) for item in entries], results_root / "experiment_index.json")
    return entries


def _extract_best_miou(metrics: dict) -> Optional[float]:
    validation = metrics.get("validation", {})
    value = validation.get("best_mean_iou")
    return float(value) if isinstance(value, (int, float)) else None


def _extract_completed_at(metrics: dict) -> Optional[str]:
    completed_at = metrics.get("completed_at")
    return str(completed_at) if completed_at is not None else None


def _extract_training_minutes(metrics: dict) -> Optional[float]:
    training = metrics.get("training", {})
    value = training.get("training_time_minutes")
    return float(value) if isinstance(value, (int, float)) else None


def _extract_trainable_params(metrics: dict) -> Optional[int]:
    model = metrics.get("model", {})
    value = model.get("trainable_params")
    return int(value) if isinstance(value, (int, float)) else None


def _isoformat_or_none(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat()
