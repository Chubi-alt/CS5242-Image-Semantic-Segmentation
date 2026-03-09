"""
Dice Coefficient calculation for semantic segmentation
"""

import numpy as np
import torch


def calculate_dice_coefficient(pred_mask, gt_mask, num_classes, ignore_index=None, smooth=1e-6):
    """
    Calculate Dice Coefficient for each class
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore (e.g., void/background)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice_per_class: Dice coefficient for each class, shape [num_classes]
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
    
    # Calculate Dice coefficient for each class
    dice_per_class = np.zeros(num_classes, dtype=np.float64)
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).astype(np.float64)
        gt_cls = (gt_mask == cls).astype(np.float64)
        
        intersection = np.sum(pred_cls * gt_cls)
        pred_sum = np.sum(pred_cls)
        gt_sum = np.sum(gt_cls)
        
        if pred_sum + gt_sum > 0:
            dice_per_class[cls] = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
        else:
            dice_per_class[cls] = np.nan  # Class not present
    
    return dice_per_class


def calculate_mean_dice(pred_mask, gt_mask, num_classes, ignore_index=None, smooth=1e-6):
    """
    Calculate Mean Dice Coefficient
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore
        smooth: Smoothing factor
    
    Returns:
        mean_dice: Mean Dice coefficient across all classes
        dice_per_class: Dice coefficient for each class
    """
    dice_per_class = calculate_dice_coefficient(
        pred_mask, gt_mask, num_classes, ignore_index, smooth
    )
    
    # Calculate mean Dice, ignoring NaN values
    valid_dice = dice_per_class[~np.isnan(dice_per_class)]
    if len(valid_dice) > 0:
        mean_dice = np.mean(valid_dice)
    else:
        mean_dice = 0.0
    
    return mean_dice, dice_per_class


def calculate_dice_batch(pred_masks, gt_masks, num_classes, ignore_index=None, smooth=1e-6):
    """
    Calculate Dice Coefficient for a batch of predictions
    
    Args:
        pred_masks: Predicted masks, shape [B, H, W], class indices
        gt_masks: Ground truth masks, shape [B, H, W], class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore
        smooth: Smoothing factor
    
    Returns:
        mean_dice: Mean Dice coefficient across all classes and all samples
        dice_per_class: Average Dice coefficient for each class across all samples
    """
    batch_size = pred_masks.shape[0]
    
    # Accumulate intersection and sums across batch
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_pred_sum = np.zeros(num_classes, dtype=np.float64)
    total_gt_sum = np.zeros(num_classes, dtype=np.float64)
    
    for i in range(batch_size):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i]
        
        # Convert to numpy if torch tensors
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        pred_mask = pred_mask.flatten()
        gt_mask = gt_mask.flatten()
        
        # Ignore specified class
        if ignore_index is not None:
            valid_mask = (gt_mask != ignore_index)
            pred_mask = pred_mask[valid_mask]
            gt_mask = gt_mask[valid_mask]
        
        for cls in range(num_classes):
            pred_cls = (pred_mask == cls).astype(np.float64)
            gt_cls = (gt_mask == cls).astype(np.float64)
            
            total_intersection[cls] += np.sum(pred_cls * gt_cls)
            total_pred_sum[cls] += np.sum(pred_cls)
            total_gt_sum[cls] += np.sum(gt_cls)
    
    # Calculate Dice coefficient for each class
    dice_per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        if total_pred_sum[cls] + total_gt_sum[cls] > 0:
            dice_per_class[cls] = (2.0 * total_intersection[cls] + smooth) / \
                                  (total_pred_sum[cls] + total_gt_sum[cls] + smooth)
        else:
            dice_per_class[cls] = np.nan
    
    # Calculate mean Dice
    valid_dice = dice_per_class[~np.isnan(dice_per_class)]
    if len(valid_dice) > 0:
        mean_dice = np.mean(valid_dice)
    else:
        mean_dice = 0.0
    
    return mean_dice, dice_per_class
