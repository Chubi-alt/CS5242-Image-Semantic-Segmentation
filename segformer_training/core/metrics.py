from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.constants import CAMVID_CLASSES, IGNORE_INDEX


def build_compute_metrics(num_labels: int = 11, ignore_index: int = IGNORE_INDEX):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, references = _unpack_eval_prediction(eval_pred)
        pred_masks, ref_masks = prepare_prediction_masks(predictions, references)
        summary, per_class_iou, _, _ = _compute_segmentation_stats(
            pred_masks,
            ref_masks,
            num_labels=num_labels,
            ignore_index=ignore_index,
        )
        metrics: Dict[str, float] = {
            "mean_iou": summary["mean_iou"],
            "mean_accuracy": summary["mean_accuracy"],
            "overall_accuracy": summary["overall_accuracy"],
        }
        for class_name, value in per_class_iou.items():
            metrics[f"iou_{class_name}"] = value
        return metrics

    return compute_metrics


def prepare_prediction_masks(predictions: np.ndarray, references: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    references_array = np.asarray(references)
    predictions_array = np.asarray(predictions)
    if predictions_array.ndim == 4:
        logits = torch.from_numpy(predictions_array)
        target_size = references_array.shape[-2:]
        if tuple(logits.shape[-2:]) != tuple(target_size):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        predictions_array = logits.argmax(dim=1).cpu().numpy()
    return predictions_array.astype(np.int64), references_array.astype(np.int64)


def compute_full_metrics(
    predictions: np.ndarray,
    references: np.ndarray,
    num_labels: int = 11,
    ignore_index: int = IGNORE_INDEX,
    compute_confusion_matrix: bool = True,
) -> Dict[str, Any]:
    pred_masks, ref_masks = prepare_prediction_masks(predictions, references)
    summary, per_class_iou, per_class_accuracy, confusion_matrix = _compute_segmentation_stats(
        pred_masks,
        ref_masks,
        num_labels=num_labels,
        ignore_index=ignore_index,
        include_confusion_matrix=compute_confusion_matrix,
    )
    payload: Dict[str, Any] = {
        "mean_iou": summary["mean_iou"],
        "mean_accuracy": summary["mean_accuracy"],
        "overall_accuracy": summary["overall_accuracy"],
        "per_class_iou": per_class_iou,
        "per_class_accuracy": per_class_accuracy,
    }
    if confusion_matrix is not None:
        payload["confusion_matrix"] = confusion_matrix.tolist()
    return payload


def build_metrics_payload(
    run_id: str,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, Any],
    parameter_counts: Dict[str, int],
    training_time_minutes: float,
    test_mode: str,
    status: str = "completed",
    backbone: str | None = None,
    continued_from: str | None = None,
    completed_at: str | None = None,
) -> Dict[str, Any]:
    validation = _strip_metric_prefix(validation_metrics, "eval_")
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "continued_from": continued_from,
        "completed_at": completed_at,
        "model": {
            "backbone": backbone,
            **parameter_counts,
        },
        "training": {
            "training_time_minutes": training_time_minutes,
        },
        "validation": {
            "mode": "trainer_eval",
            "best_mean_iou": validation.get("mean_iou"),
            "best_metrics": validation,
        },
        "test": {
            "mode": test_mode,
            **test_metrics,
        },
    }
    return payload


def _unpack_eval_prediction(eval_pred) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        return eval_pred.predictions, eval_pred.label_ids
    if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
        return eval_pred
    raise TypeError("Unsupported eval prediction payload.")


def _compute_segmentation_stats(
    predictions: np.ndarray,
    references: np.ndarray,
    num_labels: int,
    ignore_index: int,
    include_confusion_matrix: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray | None]:
    flat_predictions = predictions.reshape(-1)
    flat_references = references.reshape(-1)
    valid = flat_references != ignore_index
    flat_predictions = flat_predictions[valid]
    flat_references = flat_references[valid]

    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    if flat_references.size:
        encoded = flat_references * num_labels + flat_predictions
        confusion = np.bincount(encoded, minlength=num_labels * num_labels).reshape(num_labels, num_labels)

    true_positives = np.diag(confusion).astype(np.float64)
    predicted_total = confusion.sum(axis=0).astype(np.float64)
    reference_total = confusion.sum(axis=1).astype(np.float64)
    union = predicted_total + reference_total - true_positives

    iou = _safe_divide(true_positives, union)
    accuracy = _safe_divide(true_positives, reference_total)

    summary = {
        "mean_iou": float(np.nanmean(iou)) if np.isfinite(np.nanmean(iou)) else 0.0,
        "mean_accuracy": float(np.nanmean(accuracy)) if np.isfinite(np.nanmean(accuracy)) else 0.0,
        "overall_accuracy": float(_safe_divide(np.array([true_positives.sum()]), np.array([reference_total.sum()]))[0]),
    }
    per_class_iou = {class_name: float(iou[index]) for index, class_name in enumerate(CAMVID_CLASSES)}
    per_class_accuracy = {class_name: float(accuracy[index]) for index, class_name in enumerate(CAMVID_CLASSES)}

    normalized_confusion = None
    if include_confusion_matrix:
        normalized_confusion = np.zeros_like(confusion, dtype=np.float64)
        valid_rows = reference_total > 0
        normalized_confusion[valid_rows] = confusion[valid_rows] / reference_total[valid_rows, None]

    return summary, per_class_iou, per_class_accuracy, normalized_confusion


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator != 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _strip_metric_prefix(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    stripped: Dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        normalized_key = key[len(prefix) :] if key.startswith(prefix) else key
        stripped[normalized_key] = float(value)
    return stripped
