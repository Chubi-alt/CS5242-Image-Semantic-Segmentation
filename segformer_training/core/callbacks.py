from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.constants import TRAINING_HISTORY_HEADERS
from core.utils import atomic_write_csv, read_csv_rows, utc_timestamp, write_status_json

try:  # pragma: no cover
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    class TrainerCallback:  # type: ignore[override]
        pass


class LocalMetricsCallback(TrainerCallback):
    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.rows: List[Dict[str, Any]] = read_csv_rows(self.output_path)
        self.best_eval_mean_iou: Optional[float] = _find_best_miou(self.rows)
        self._seen_events = {_row_key(row) for row in self.rows}

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args
        del control
        del kwargs
        row = _empty_row()
        row.update(
            {
                "timestamp": utc_timestamp(),
                "step": getattr(state, "global_step", ""),
                "epoch": _format_epoch(getattr(state, "epoch", None)),
            }
        )
        logs = logs or {}
        if any(key.startswith("eval_") for key in logs):
            row["event"] = "eval"
            row["eval_loss"] = logs.get("eval_loss", "")
            row["eval_mean_iou"] = logs.get("eval_mean_iou", "")
            row["eval_overall_accuracy"] = logs.get("eval_overall_accuracy", "")
            row["lr"] = logs.get("learning_rate", logs.get("lr", ""))
            best_so_far = 0
            eval_mean_iou = logs.get("eval_mean_iou")
            if isinstance(eval_mean_iou, (int, float)):
                if self.best_eval_mean_iou is None or eval_mean_iou > self.best_eval_mean_iou:
                    self.best_eval_mean_iou = float(eval_mean_iou)
                    best_so_far = 1
            row["best_so_far"] = best_so_far
        else:
            row["event"] = "log"
            row["train_loss"] = logs.get("loss", logs.get("train_loss", ""))
            row["lr"] = logs.get("learning_rate", logs.get("lr", ""))
        self._append_row(row)

    def on_save(self, args, state, control, **kwargs):
        del control
        del kwargs
        row = _empty_row()
        row.update(
            {
                "timestamp": utc_timestamp(),
                "step": getattr(state, "global_step", ""),
                "epoch": _format_epoch(getattr(state, "epoch", None)),
                "event": "save",
                "checkpoint_path": str(Path(args.output_dir) / f"checkpoint-{getattr(state, 'global_step', 0)}"),
            }
        )
        self._append_row(row)

    def on_train_end(self, args, state, control, **kwargs):
        del args
        del control
        del kwargs
        row = _empty_row()
        row.update(
            {
                "timestamp": utc_timestamp(),
                "step": getattr(state, "global_step", ""),
                "epoch": _format_epoch(getattr(state, "epoch", None)),
                "event": "train_end",
            }
        )
        self._append_row(row)

    def _append_row(self, row: Dict[str, Any]) -> None:
        key = _row_key(row)
        if key in self._seen_events:
            return
        self.rows.append(row)
        self._seen_events.add(key)
        atomic_write_csv(self.rows, self.output_path, TRAINING_HISTORY_HEADERS)


class CheckpointMarkerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        del control
        del kwargs
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{getattr(state, 'global_step', 0)}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / ".save_complete").write_text("", encoding="utf-8")


def _empty_row() -> Dict[str, Any]:
    return {key: "" for key in TRAINING_HISTORY_HEADERS}


def _format_epoch(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _find_best_miou(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = []
    for row in rows:
        raw = row.get("eval_mean_iou")
        if raw in ("", None):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("step", "")),
        str(row.get("event", "")),
        str(row.get("checkpoint_path", "")),
    )


import collections
import os
import socket
import time


class StatusCallback(TrainerCallback):
    """Writes status.json heartbeat every N steps. Write failures are non-fatal."""

    def __init__(
        self,
        run_dir: str,
        run_id: str,
        experiment_name: str,
        total_epochs: int,
        total_steps: int,
        steps_per_epoch: int,
        steps_per_epoch_source: str = "exact",
        hostname: str = "",
        pid: int = 0,
        device: str = "unknown",
        heartbeat_steps: int = 5,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch_source = steps_per_epoch_source
        self.hostname = hostname or socket.gethostname()
        self.pid = pid or os.getpid()
        self.device = device
        self.heartbeat_steps = heartbeat_steps
        self._train_loss: Optional[float] = None
        self._last_eval_miou: Optional[float] = None
        self._best_eval_miou: Optional[float] = None
        # Sliding window for ETA: (monotonic_time, global_step)
        self._step_times: collections.deque = collections.deque(maxlen=20)

    def on_step_end(self, args, state, control, **kwargs) -> None:  # type: ignore[override]
        step = getattr(state, "global_step", 0)
        if step % self.heartbeat_steps != 0:
            return
        self._step_times.append((time.monotonic(), step))
        eta = self._estimate_eta(step)
        epoch = int(getattr(state, "epoch", 0) or 0)
        epoch_step = step - epoch * self.steps_per_epoch if self.steps_per_epoch > 0 else None
        write_status_json(
            run_dir=self.run_dir,
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            phase="training",
            phase_step=epoch_step,
            phase_total_steps=self.steps_per_epoch,
            phase_message=f"epoch {epoch}/{self.total_epochs} · step {step}/{self.total_steps}",
            current_epoch=epoch,
            total_epochs=self.total_epochs,
            global_step=step,
            total_steps=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            steps_per_epoch_source=self.steps_per_epoch_source,
            train_loss=self._train_loss,
            last_eval_miou=self._last_eval_miou,
            best_eval_miou=self._best_eval_miou,
            eta_seconds=eta,
            eta_source="training" if eta is not None else None,
            hostname=self.hostname,
            pid=self.pid,
            device=self.device,
        )

    def on_evaluate(self, args, state, control, **kwargs) -> None:  # type: ignore[override]
        step = getattr(state, "global_step", 0)
        epoch = int(getattr(state, "epoch", 0) or 0)
        write_status_json(
            run_dir=self.run_dir,
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            phase="validating",
            phase_step=None,
            phase_total_steps=None,
            phase_message="running validation...",
            current_epoch=epoch,
            total_epochs=self.total_epochs,
            global_step=step,
            total_steps=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            steps_per_epoch_source=self.steps_per_epoch_source,
            train_loss=self._train_loss,
            last_eval_miou=self._last_eval_miou,
            best_eval_miou=self._best_eval_miou,
            eta_seconds=None,
            eta_source=None,
            hostname=self.hostname,
            pid=self.pid,
            device=self.device,
        )

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # type: ignore[override]
        logs = logs or {}
        if "loss" in logs or "train_loss" in logs:
            self._train_loss = float(logs.get("loss", logs.get("train_loss", self._train_loss) or 0))
        miou = logs.get("eval_mean_iou")
        if isinstance(miou, (int, float)):
            self._last_eval_miou = float(miou)
            if self._best_eval_miou is None or float(miou) > self._best_eval_miou:
                self._best_eval_miou = float(miou)
            # Write updated miou to status
            step = getattr(state, "global_step", 0)
            epoch = int(getattr(state, "epoch", 0) or 0)
            write_status_json(
                run_dir=self.run_dir,
                run_id=self.run_id,
                experiment_name=self.experiment_name,
                phase="training",
                phase_step=None,
                phase_total_steps=self.steps_per_epoch,
                phase_message=f"epoch {epoch}/{self.total_epochs} · eval done",
                current_epoch=epoch,
                total_epochs=self.total_epochs,
                global_step=step,
                total_steps=self.total_steps,
                steps_per_epoch=self.steps_per_epoch,
                steps_per_epoch_source=self.steps_per_epoch_source,
                train_loss=self._train_loss,
                last_eval_miou=self._last_eval_miou,
                best_eval_miou=self._best_eval_miou,
                eta_seconds=None,
                eta_source=None,
                hostname=self.hostname,
                pid=self.pid,
                device=self.device,
            )

    def _estimate_eta(self, current_step: int) -> Optional[float]:
        if len(self._step_times) < 2:
            return None
        t0, s0 = self._step_times[0]
        t1, s1 = self._step_times[-1]
        if s1 <= s0:
            return None
        rate = (t1 - t0) / (s1 - s0)  # seconds per step
        remaining = self.total_steps - current_step
        return rate * remaining if remaining > 0 else 0.0
