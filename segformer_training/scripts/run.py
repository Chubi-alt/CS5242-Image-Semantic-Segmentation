from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import warnings

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.callbacks import CheckpointMarkerCallback, LocalMetricsCallback, StatusCallback
from core.config import ConfigValidationError, load_config, resolve_run_context, save_config_snapshot
from core.dataset import build_datasets
from core.index import ExperimentIndexEntry, load_experiment_index, update_experiment_index
from core.losses import build_loss_fn
from core.metrics import build_compute_metrics, build_metrics_payload, compute_full_metrics, prepare_prediction_masks
from core.distillation import build_kd_loss
from core.model import build_model, count_parameters, export_inference_bundle
from core.trainer import SegFormerTrainer, build_training_args
from core.tta import tta_evaluate
from core.utils import (
    atomic_write_json,
    find_valid_checkpoint,
    parse_run_id_from_export_path,
    preflight_checks,
    read_csv_rows,
    resolve_report_to,
    set_seed,
    utc_timestamp,
    write_status_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single SegExpress experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values with key=value.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous run.")
    parser.add_argument("--run-id", default=None, help="Exact run id to resume.")
    args = parser.parse_args(argv)

    run_context = None
    config = None
    results_dir = "results"
    started_at = None
    continued_from = None
    snapshot_path = None
    try:
        config = load_config(args.config, overrides=args.overrides)

        # Resolve results_dir and checkpoints_dir from config.paths, relative to config file
        _config_path = Path(args.config).resolve()
        _results_dir = str(config.paths.resolve_results_dir(_config_path))
        _checkpoints_dir = config.paths.resolve_checkpoints_dir(_config_path)
        _checkpoints_dir_str = str(_checkpoints_dir) if _checkpoints_dir is not None else None
        _data_dir = str(config.data.resolve_data_dir(_config_path))

        run_context = resolve_run_context(
            config,
            resume=args.resume,
            run_id=args.run_id,
            results_dir=_results_dir,
            checkpoints_dir=_checkpoints_dir_str,
        )
        preflight_checks(config, run_context, data_dir=_data_dir)
        set_seed(config.experiment.seed)
        report_to = resolve_report_to(config)

        run_dir = Path(run_context.run_dir)
        checkpoint_dir = Path(run_context.checkpoint_dir)
        results_dir = _results_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_status_json(
            run_dir=run_context.run_dir, run_id=run_context.run_id,
            experiment_name=config.experiment.name,
            phase="initialized", phase_step=None, phase_total_steps=None,
            phase_message="preflight checks passed",
            current_epoch=0, total_epochs=config.training.num_train_epochs,
            global_step=0, total_steps=0, steps_per_epoch=0,
            steps_per_epoch_source="unknown",
            train_loss=None, last_eval_miou=None, best_eval_miou=None,
            eta_seconds=None, eta_source=None,
            hostname=socket.gethostname(), pid=os.getpid(), device="unknown",
        )

        if run_context.resumed:
            snapshot_path = str(run_dir / "config_snapshot.yaml")
        else:
            snapshot_path = save_config_snapshot(config, run_context.run_dir)

        started_at = _resolve_started_at(run_context.run_id, results_dir) or utc_timestamp()
        continued_from = None if run_context.resumed else _resolve_continued_from(config)

        if not run_context.resumed:
            update_experiment_index(
                ExperimentIndexEntry(
                    run_id=run_context.run_id,
                    experiment_name=config.experiment.name,
                    status="initialized",
                    tags=list(config.experiment.tags),
                    config_path=str(Path(args.config)),
                    snapshot_path=snapshot_path,
                    best_miou=None,
                    best_checkpoint=None,
                    started_at=started_at,
                    completed_at=None,
                    continued_from=continued_from,
                    latest_requested_epochs=int(config.training.num_train_epochs),
                    training_time_minutes=None,
                    trainable_params=None,
                    backbone=config.model.backbone,
                ),
                results_dir=results_dir,
            )

        update_experiment_index(
            ExperimentIndexEntry(
                run_id=run_context.run_id,
                experiment_name=config.experiment.name,
                status="training",
                tags=list(config.experiment.tags),
                config_path=str(Path(args.config)),
                snapshot_path=snapshot_path,
                best_miou=None,
                best_checkpoint=None,
                started_at=started_at,
                completed_at=None,
                continued_from=continued_from,
                latest_requested_epochs=int(config.training.num_train_epochs),
                training_time_minutes=None,
                trainable_params=None,
                backbone=config.model.backbone,
            ),
            results_dir=results_dir,
        )

        model, processor = build_model(config.model)

        # ── Knowledge Distillation setup ──────────────────────────────────────
        teacher_model = None
        kd_loss_fn = None
        if config.kd.enabled:
            teacher_cfg = _build_teacher_model_config(config)
            teacher_model, _ = build_model(teacher_cfg)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            model, kd_loss_fn = build_kd_loss(
                config.kd, model, teacher_model,
                ignore_index=config.data.ignore_index,
            )
            # model may now be SegFormerWithAdaptors (method=multilevel)
        # ─────────────────────────────────────────────────────────────────────

        parameter_counts = count_parameters(model)
        bundle = build_datasets(config, processor, data_dir=_data_dir)
        loss_fn = build_loss_fn(
            config.loss,
            num_classes=config.model.num_labels,
            ignore_index=config.data.ignore_index,
            class_frequencies=bundle.class_frequencies,
        )
        training_args = build_training_args(config.training, run_context, report_to)

        import torch as _torch
        _device_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "cpu"
        _steps_per_epoch = max(1, len(bundle.train_dataset) // config.training.per_device_train_batch_size)
        _total_steps = _steps_per_epoch * config.training.num_train_epochs
        _steps_source = "exact"

        status_cb = StatusCallback(
            run_dir=str(run_dir),
            run_id=run_context.run_id,
            experiment_name=config.experiment.name,
            total_epochs=config.training.num_train_epochs,
            total_steps=_total_steps,
            steps_per_epoch=_steps_per_epoch,
            steps_per_epoch_source=_steps_source,
            device=_device_name,
        )

        callbacks = [
            LocalMetricsCallback(str(run_dir / "training_history.csv")),
            CheckpointMarkerCallback(),
            status_cb,
        ]
        trainer = SegFormerTrainer(
            model=model,
            args=training_args,
            train_dataset=bundle.train_dataset,
            eval_dataset=bundle.val_dataset,
            compute_metrics=build_compute_metrics(
                num_labels=config.model.num_labels,
                ignore_index=config.data.ignore_index,
            ),
            callbacks=callbacks,
            loss_fn=loss_fn,
            aux_weight=config.model.auxiliary_heads.weight,
            teacher_model=teacher_model,
            kd_loss_fn=kd_loss_fn,
        )

        train_start = time.monotonic()
        train_kwargs = {}
        if run_context.resumed and run_context.resume_from_checkpoint is not None:
            train_kwargs["resume_from_checkpoint"] = run_context.resume_from_checkpoint
        trainer.train(**train_kwargs)
        training_time_minutes = (time.monotonic() - train_start) / 60.0

        validation_metrics = trainer.evaluate(eval_dataset=bundle.val_dataset)
        latest_checkpoint = find_valid_checkpoint(checkpoint_dir)
        best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None) or latest_checkpoint
        if latest_checkpoint is None:
            raise RuntimeError(f"No valid checkpoint was produced under {checkpoint_dir}")

        best_dir = checkpoint_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        last_dir = checkpoint_dir / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json({"checkpoint_path": best_checkpoint}, best_dir / "pointer.json")
        atomic_write_json({"checkpoint_path": latest_checkpoint}, last_dir / "pointer.json")
        # Use _save to correctly handle SegFormerWithAuxHeads / SegFormerWithAdaptors
        trainer._save(str(best_dir))
        export_inference_bundle(trainer.model, processor, best_dir / "segformer")

        evaluation_dataset = bundle.test_dataset or bundle.val_dataset
        if evaluation_dataset is None:
            raise RuntimeError("No evaluation dataset available.")

        update_experiment_index(
            ExperimentIndexEntry(
                run_id=run_context.run_id,
                experiment_name=config.experiment.name,
                status="testing",
                tags=list(config.experiment.tags),
                config_path=str(Path(args.config)),
                snapshot_path=snapshot_path,
                best_miou=None,
                best_checkpoint=str(best_dir),
                started_at=started_at,
                completed_at=None,
                continued_from=continued_from,
                latest_requested_epochs=int(config.training.num_train_epochs),
                training_time_minutes=training_time_minutes,
                trainable_params=parameter_counts["trainable_params"],
                backbone=config.model.backbone,
            ),
            results_dir=results_dir,
        )
        _eval_total = len(evaluation_dataset)
        write_status_json(
            run_dir=str(run_dir), run_id=run_context.run_id,
            experiment_name=config.experiment.name,
            phase="testing", phase_step=0, phase_total_steps=_eval_total,
            phase_message=f"0/{_eval_total} images",
            current_epoch=config.training.num_train_epochs,
            total_epochs=config.training.num_train_epochs,
            global_step=_total_steps, total_steps=_total_steps,
            steps_per_epoch=_steps_per_epoch, steps_per_epoch_source=_steps_source,
            train_loss=status_cb._train_loss, last_eval_miou=status_cb._last_eval_miou,
            best_eval_miou=status_cb._best_eval_miou,
            eta_seconds=None, eta_source=None,
            hostname=socket.gethostname(), pid=os.getpid(), device=_device_name,
        )

        def _write_test_progress(step: int, total: int) -> None:
            write_status_json(
                run_dir=str(run_dir), run_id=run_context.run_id,
                experiment_name=config.experiment.name,
                phase="testing", phase_step=step, phase_total_steps=total,
                phase_message=f"{step}/{total} images",
                current_epoch=config.training.num_train_epochs,
                total_epochs=config.training.num_train_epochs,
                global_step=_total_steps, total_steps=_total_steps,
                steps_per_epoch=_steps_per_epoch, steps_per_epoch_source=_steps_source,
                train_loss=status_cb._train_loss, last_eval_miou=status_cb._last_eval_miou,
                best_eval_miou=status_cb._best_eval_miou,
                eta_seconds=None, eta_source=None,
                hostname=socket.gethostname(), pid=os.getpid(), device=_device_name,
            )

        test_mode, test_metrics, prediction_triplets = _run_final_evaluation(
            trainer=trainer,
            processor=processor,
            dataset=evaluation_dataset,
            config=config,
            step_callback=_write_test_progress,
        )

        trained_metrics = build_metrics_payload(
            run_id=run_context.run_id,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            parameter_counts=parameter_counts,
            training_time_minutes=training_time_minutes,
            test_mode=test_mode,
            status="trained",
            backbone=config.model.backbone,
            continued_from=continued_from,
            completed_at=None,
        )
        trained_metrics["training"]["epochs_completed"] = int(config.training.num_train_epochs)
        metrics_path = run_dir / "metrics.json"
        atomic_write_json(trained_metrics, metrics_path)

        update_experiment_index(
            ExperimentIndexEntry(
                run_id=run_context.run_id,
                experiment_name=config.experiment.name,
                status="trained",
                tags=list(config.experiment.tags),
                config_path=str(Path(args.config)),
                snapshot_path=snapshot_path,
                best_miou=float(trained_metrics["validation"]["best_mean_iou"]) if trained_metrics["validation"]["best_mean_iou"] is not None else None,
                best_checkpoint=str(best_dir),
                started_at=started_at,
                completed_at=None,
                continued_from=continued_from,
                latest_requested_epochs=int(config.training.num_train_epochs),
                training_time_minutes=training_time_minutes,
                trainable_params=parameter_counts["trainable_params"],
                backbone=config.model.backbone,
            ),
            results_dir=results_dir,
        )

        completed_at = utc_timestamp()
        completed_metrics = build_metrics_payload(
            run_id=run_context.run_id,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            parameter_counts=parameter_counts,
            training_time_minutes=training_time_minutes,
            test_mode=test_mode,
            status="completed",
            backbone=config.model.backbone,
            continued_from=continued_from,
            completed_at=completed_at,
        )
        completed_metrics["training"]["epochs_completed"] = int(config.training.num_train_epochs)
        atomic_write_json(completed_metrics, metrics_path)

        update_experiment_index(
            ExperimentIndexEntry(
                run_id=run_context.run_id,
                experiment_name=config.experiment.name,
                status="completed",
                tags=list(config.experiment.tags),
                config_path=str(Path(args.config)),
                snapshot_path=snapshot_path,
                best_miou=float(completed_metrics["validation"]["best_mean_iou"]) if completed_metrics["validation"]["best_mean_iou"] is not None else None,
                best_checkpoint=str(best_dir),
                started_at=started_at,
                completed_at=completed_at,
                continued_from=continued_from,
                latest_requested_epochs=int(config.training.num_train_epochs),
                training_time_minutes=training_time_minutes,
                trainable_params=parameter_counts["trainable_params"],
                backbone=config.model.backbone,
            ),
            results_dir=results_dir,
        )
        best_miou_val = completed_metrics['validation']['best_mean_iou']
        best_miou_str = f"{best_miou_val:.3f}" if best_miou_val is not None else "n/a"
        write_status_json(
            run_dir=str(run_dir), run_id=run_context.run_id,
            experiment_name=config.experiment.name,
            phase="completed", phase_step=None, phase_total_steps=None,
            phase_message=f"best mIoU {best_miou_str} · {config.training.num_train_epochs} epochs",
            current_epoch=config.training.num_train_epochs,
            total_epochs=config.training.num_train_epochs,
            global_step=_total_steps, total_steps=_total_steps,
            steps_per_epoch=_steps_per_epoch, steps_per_epoch_source=_steps_source,
            train_loss=status_cb._train_loss, last_eval_miou=status_cb._last_eval_miou,
            best_eval_miou=status_cb._best_eval_miou,
            eta_seconds=0.0, eta_source=None,
            hostname=socket.gethostname(), pid=os.getpid(), device=_device_name,
        )
        print(f"Run completed: {run_context.run_id}")
        print(f"Artifacts: {run_context.run_dir}")
        return 0
    except (ConfigValidationError, RuntimeError, FileNotFoundError, ValueError, NotImplementedError) as exc:
        _mark_crashed(
            run_context=run_context,
            config=config,
            config_path=getattr(args, "config", None),
            results_dir=results_dir,
            started_at=started_at,
            continued_from=continued_from,
            snapshot_path=snapshot_path,
        )
        try:
            if run_context is not None:
                write_status_json(
                    run_dir=run_context.run_dir, run_id=run_context.run_id,
                    experiment_name=getattr(config, "experiment", None) and config.experiment.name or "unknown",
                    phase="crashed", phase_step=None, phase_total_steps=None,
                    phase_message=str(exc)[:200],
                    current_epoch=0, total_epochs=0, global_step=0, total_steps=0,
                    steps_per_epoch=0, steps_per_epoch_source="unknown",
                    train_loss=None, last_eval_miou=None, best_eval_miou=None,
                    eta_seconds=None, eta_source=None,
                    hostname=socket.gethostname(), pid=os.getpid(), device="unknown",
                )
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        _mark_crashed(
            run_context=run_context,
            config=config,
            config_path=getattr(args, "config", None),
            results_dir=results_dir,
            started_at=started_at,
            continued_from=continued_from,
            snapshot_path=snapshot_path,
        )
        try:
            if run_context is not None:
                write_status_json(
                    run_dir=run_context.run_dir, run_id=run_context.run_id,
                    experiment_name=getattr(config, "experiment", None) and config.experiment.name or "unknown",
                    phase="crashed", phase_step=None, phase_total_steps=None,
                    phase_message="unexpected error — see logs",
                    current_epoch=0, total_epochs=0, global_step=0, total_steps=0,
                    steps_per_epoch=0, steps_per_epoch_source="unknown",
                    train_loss=None, last_eval_miou=None, best_eval_miou=None,
                    eta_seconds=None, eta_source=None,
                    hostname=socket.gethostname(), pid=os.getpid(), device="unknown",
                )
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _run_final_evaluation(trainer, processor, dataset, config, step_callback=None):
    device = _infer_model_device(trainer.model)
    limit = config.evaluation.num_visualizations
    if config.evaluation.tta.enabled:
        test_metrics, prediction_triplets = tta_evaluate(
            model=trainer.model,
            processor=processor,
            test_dataset=dataset,
            config=config.evaluation.tta,
            device=device,
            step_callback=step_callback,
            num_labels=config.model.num_labels,
            ignore_index=config.data.ignore_index,
            compute_confusion_matrix=config.evaluation.compute_confusion_matrix,
            prediction_limit=limit,
        )
        return "tta", test_metrics, prediction_triplets

    prediction_output = _single_scale_predict_with_progress(
        trainer, dataset, config, step_callback=step_callback
    )
    prediction_masks, reference_masks = prepare_prediction_masks(
        prediction_output.predictions,
        prediction_output.label_ids,
    )
    test_metrics = compute_full_metrics(
        prediction_masks,
        reference_masks,
        num_labels=config.model.num_labels,
        ignore_index=config.data.ignore_index,
        compute_confusion_matrix=config.evaluation.compute_confusion_matrix,
    )
    prediction_triplets = _build_prediction_triplets(dataset, prediction_masks, limit)
    return "single_scale", test_metrics, prediction_triplets


def _build_prediction_triplets(dataset, prediction_masks, limit: int):
    max_items = len(dataset) if limit == -1 else min(len(dataset), max(limit, 0))
    triplets = []
    for index in range(max_items):
        raw_sample = dataset.get_raw_sample(index)
        triplets.append((raw_sample.image, raw_sample.label, prediction_masks[index]))
    return triplets



def _infer_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _single_scale_predict_with_progress(trainer, dataset, config, step_callback=None):
    """Run single-scale inference with optional per-batch progress reporting.

    Replaces trainer.predict() so that step_callback can be called after each
    batch, giving real-time status.json updates during the testing phase.
    """
    from types import SimpleNamespace

    model = trainer.model
    model.eval()
    device = _infer_model_device(model)
    batch_size = max(1, config.training.per_device_eval_batch_size)
    total = len(dataset)
    all_logits = []
    all_labels = []
    use_bf16 = config.training.bf16 and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = [dataset[i] for i in range(start, end)]
            pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)
            labels = torch.stack([item["labels"] for item in batch])
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            if logits.dtype != torch.float32:
                logits = logits.float()
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())
            if step_callback is not None:
                step_callback(end, total)

    return SimpleNamespace(
        predictions=np.concatenate(all_logits, axis=0),
        label_ids=np.concatenate(all_labels, axis=0),
    )


def _build_teacher_model_config(config):
    """Create a ModelConfig for the frozen B2 teacher from the KD config."""
    import copy
    teacher_cfg = copy.deepcopy(config.model)
    teacher_cfg.backbone = "nvidia/mit-b2"
    teacher_cfg.resume_from = config.kd.teacher_checkpoint
    teacher_cfg.auxiliary_heads.enabled = False
    return teacher_cfg


def _resolve_continued_from(config) -> str | None:
    if config.model.resume_from is None:
        return None
    return parse_run_id_from_export_path(config.model.resume_from)


def _resolve_started_at(run_id: str, results_dir: str) -> str | None:
    for entry in load_experiment_index(results_dir=results_dir):
        if entry.run_id == run_id:
            return entry.started_at
    return None


def _mark_crashed(run_context, config, config_path, results_dir, started_at, continued_from, snapshot_path) -> None:
    if run_context is None or config is None or config_path is None:
        return
    update_experiment_index(
        ExperimentIndexEntry(
            run_id=run_context.run_id,
            experiment_name=config.experiment.name,
            status="crashed",
            tags=list(config.experiment.tags),
            config_path=str(Path(config_path)),
            snapshot_path=str(snapshot_path or Path(run_context.run_dir) / "config_snapshot.yaml"),
            best_miou=None,
            best_checkpoint=None,
            started_at=started_at or utc_timestamp(),
            completed_at=None,
            continued_from=continued_from,
            latest_requested_epochs=int(config.training.num_train_epochs),
            training_time_minutes=None,
            trainable_params=None,
            backbone=config.model.backbone,
        ),
        results_dir=results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
