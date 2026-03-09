"""
Standardized Mean Intersection over Union (mIoU) calculation
"""

import numpy as np
import torch


def calculate_iou(pred_mask, gt_mask, num_classes, ignore_index=None):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore (e.g., void/background)
    
    Returns:
        iou_per_class: IoU for each class, shape [num_classes]
        intersection: Intersection for each class, shape [num_classes]
        union: Union for each class, shape [num_classes]
    """
    # Convert to numpy if torch tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Flatten if batch dimension exists
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.flatten()
        gt_mask = gt_mask.flatten()
    elif pred_mask.ndim == 2:
        pred_mask = pred_mask.flatten()
        gt_mask = gt_mask.flatten()
    
    # Ignore specified class
    if ignore_index is not None:
        valid_mask = (gt_mask != ignore_index)
        pred_mask = pred_mask[valid_mask]
        gt_mask = gt_mask[valid_mask]
    
    # Calculate intersection and union for each class
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection[cls] = np.logical_and(pred_cls, gt_cls).sum()
        union[cls] = np.logical_or(pred_cls, gt_cls).sum()
    
    # Calculate IoU for each class
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        if union[cls] > 0:
            iou_per_class[cls] = intersection[cls] / union[cls]
        else:
            iou_per_class[cls] = np.nan  # Class not present in ground truth
    
    return iou_per_class, intersection, union


def calculate_miou(pred_mask, gt_mask, num_classes, ignore_index=None):
    """
    Calculate Mean Intersection over Union (mIoU)
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore (e.g., void/background)
    
    Returns:
        miou: Mean IoU across all classes (excluding NaN classes)
        iou_per_class: IoU for each class
    """
    iou_per_class, _, _ = calculate_iou(pred_mask, gt_mask, num_classes, ignore_index)
    
    # Calculate mean IoU, ignoring NaN values (classes not present)
    valid_ious = iou_per_class[~np.isnan(iou_per_class)]
    if len(valid_ious) > 0:
        miou = np.mean(valid_ious)
    else:
        miou = 0.0
    
    return miou, iou_per_class


def calculate_miou_batch(pred_masks, gt_masks, num_classes, ignore_index=None):
    """
    Calculate mIoU for a batch of predictions
    
    Args:
        pred_masks: Predicted masks, shape [B, H, W], class indices
        gt_masks: Ground truth masks, shape [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        miou: Mean IoU across all classes and all samples
        iou_per_class: Average IoU for each class across all samples
    """
    batch_size = pred_masks.shape[0]
    
    # Accumulate intersection and union across batch
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    
    for i in range(batch_size):
        iou_per_class, intersection, union = calculate_iou(
            pred_masks[i], gt_masks[i], num_classes, ignore_index
        )
        total_intersection += intersection
        total_union += union
    
    # Calculate IoU for each class across all samples
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        if total_union[cls] > 0:
            iou_per_class[cls] = total_intersection[cls] / total_union[cls]
        else:
            iou_per_class[cls] = np.nan
    
    # Calculate mean IoU
    valid_ious = iou_per_class[~np.isnan(iou_per_class)]
    if len(valid_ious) > 0:
        miou = np.mean(valid_ious)
    else:
        miou = 0.0
    
    return miou, iou_per_class
