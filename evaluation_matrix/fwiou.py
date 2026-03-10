"""
Frequency Weighted Intersection over Union (FWIoU) calculation
Weighted by class frequency to handle class imbalance in road scenes
"""

import numpy as np
import torch
from miou import calculate_iou


def calculate_fwiou(pred_mask, gt_mask, num_classes, ignore_index=None):
    """
    Calculate Frequency Weighted Intersection over Union (FWIoU)
    
    FWIoU weights each class's IoU by its frequency (pixel proportion) in the ground truth.
    This gives more weight to classes that appear more frequently, helping balance
    the evaluation when there are large background classes and small foreground classes.
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore (e.g., void/background)
    
    Returns:
        fwiou: Frequency Weighted IoU
        iou_per_class: IoU for each class
        weights: Frequency weights for each class
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
    
    # Calculate IoU for each class
    iou_per_class, _, union = calculate_iou(pred_mask, gt_mask, num_classes, ignore_index)
    
    # Calculate frequency weights (proportion of pixels for each class in ground truth)
    total_pixels = len(gt_mask)
    weights = np.zeros(num_classes, dtype=np.float64)
    
    for cls in range(num_classes):
        gt_cls_count = np.sum(gt_mask == cls)
        if total_pixels > 0:
            weights[cls] = gt_cls_count / total_pixels
        else:
            weights[cls] = 0.0
    
    # Calculate weighted IoU
    # Only consider classes that are present (non-NaN IoU and non-zero weight)
    valid_mask = ~np.isnan(iou_per_class) & (weights > 0)
    if np.any(valid_mask):
        # Normalize weights to sum to 1 for valid classes
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / valid_weights.sum()
        
        # Calculate weighted average
        fwiou = np.sum(iou_per_class[valid_mask] * valid_weights)
    else:
        fwiou = 0.0
    
    return fwiou, iou_per_class, weights


def calculate_fwiou_batch(pred_masks, gt_masks, num_classes, ignore_index=None):
    """
    Calculate Frequency Weighted IoU for a batch of predictions
    
    Args:
        pred_masks: Predicted masks, shape [B, H, W], class indices
        gt_masks: Ground truth masks, shape [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        fwiou: Frequency Weighted IoU across all samples
        iou_per_class: Average IoU for each class across all samples
        weights: Average frequency weights for each class across all samples
    """
    batch_size = pred_masks.shape[0]
    
    # Accumulate intersection and union across batch
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    total_class_pixels = np.zeros(num_classes, dtype=np.float64)
    
    for i in range(batch_size):
        iou_per_class, intersection, union = calculate_iou(
            pred_masks[i], gt_masks[i], num_classes, ignore_index
        )
        total_intersection += intersection
        total_union += union
        
        # Calculate class frequencies for this sample
        gt_mask = gt_masks[i]
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        gt_mask = gt_mask.flatten()
        
        if ignore_index is not None:
            valid_mask = (gt_mask != ignore_index)
            gt_mask = gt_mask[valid_mask]
        
        total_pixels += len(gt_mask)
        for cls in range(num_classes):
            total_class_pixels[cls] += np.sum(gt_mask == cls)
    
    # Calculate IoU for each class across all samples
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        if total_union[cls] > 0:
            iou_per_class[cls] = total_intersection[cls] / total_union[cls]
        else:
            iou_per_class[cls] = np.nan
    
    # Calculate frequency weights across all samples
    weights = np.zeros(num_classes, dtype=np.float64)
    if total_pixels > 0:
        weights = total_class_pixels / total_pixels
    
    # Calculate weighted IoU
    valid_mask = ~np.isnan(iou_per_class) & (weights > 0)
    if np.any(valid_mask):
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / valid_weights.sum()
        fwiou = np.sum(iou_per_class[valid_mask] * valid_weights)
    else:
        fwiou = 0.0
    
    return fwiou, iou_per_class, weights
