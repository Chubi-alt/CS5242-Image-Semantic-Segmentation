"""
Pixel Accuracy calculation for semantic segmentation
"""

import numpy as np
import torch


def calculate_pixel_accuracy(pred_mask, gt_mask, ignore_index=None):
    """
    Calculate Pixel Accuracy
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        ignore_index: Class index to ignore (e.g., void/background)
    
    Returns:
        pixel_acc: Pixel accuracy (percentage of correctly predicted pixels)
        correct_pixels: Number of correctly predicted pixels
        total_pixels: Total number of pixels (excluding ignored pixels)
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
    
    # Calculate pixel accuracy
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = len(pred_mask)
    
    if total_pixels > 0:
        pixel_acc = correct_pixels / total_pixels
    else:
        pixel_acc = 0.0
    
    return pixel_acc, correct_pixels, total_pixels


def calculate_pixel_accuracy_batch(pred_masks, gt_masks, ignore_index=None):
    """
    Calculate Pixel Accuracy for a batch of predictions
    
    Args:
        pred_masks: Predicted masks, shape [B, H, W], class indices
        gt_masks: Ground truth masks, shape [B, H, W], class indices
        ignore_index: Class index to ignore
    
    Returns:
        pixel_acc: Pixel accuracy across all samples
        correct_pixels: Total number of correctly predicted pixels
        total_pixels: Total number of pixels (excluding ignored pixels)
    """
    batch_size = pred_masks.shape[0]
    
    total_correct = 0
    total_pixels = 0
    
    for i in range(batch_size):
        pixel_acc, correct, total = calculate_pixel_accuracy(
            pred_masks[i], gt_masks[i], ignore_index
        )
        total_correct += correct
        total_pixels += total
    
    if total_pixels > 0:
        pixel_acc = total_correct / total_pixels
    else:
        pixel_acc = 0.0
    
    return pixel_acc, total_correct, total_pixels


def calculate_class_accuracy(pred_mask, gt_mask, num_classes, ignore_index=None):
    """
    Calculate per-class pixel accuracy
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        class_acc: Pixel accuracy for each class, shape [num_classes]
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
    
    # Calculate accuracy for each class
    class_acc = np.zeros(num_classes, dtype=np.float64)
    
    for cls in range(num_classes):
        gt_cls_mask = (gt_mask == cls)
        if np.sum(gt_cls_mask) > 0:
            pred_cls = (pred_mask == cls)
            correct = np.sum(np.logical_and(gt_cls_mask, pred_cls))
            total = np.sum(gt_cls_mask)
            class_acc[cls] = correct / total
        else:
            class_acc[cls] = np.nan  # Class not present in ground truth
    
    return class_acc
