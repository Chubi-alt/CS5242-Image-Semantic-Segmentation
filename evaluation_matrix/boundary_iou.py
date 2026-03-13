"""
Boundary Intersection over Union (Boundary IoU) calculation
Evaluates segmentation quality specifically at object boundaries/edges
"""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, binary_dilation
from evaluation_matrix.miou import calculate_iou


def extract_boundary(mask, boundary_width=1):
    """
    Extract boundary pixels from a mask
    
    Args:
        mask: Binary mask or class mask, shape [H, W]
        boundary_width: Width of boundary region in pixels (default: 1)
    
    Returns:
        boundary_mask: Binary mask indicating boundary pixels
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # For each class, extract its boundary
    boundary_mask = np.zeros_like(mask, dtype=bool)
    
    # Get unique classes in mask
    unique_classes = np.unique(mask)
    
    for cls in unique_classes:
        # Create binary mask for this class
        class_mask = (mask == cls).astype(np.uint8)
        
        if np.sum(class_mask) == 0:
            continue
        
        # Use morphological operations to find boundaries
        # Erode the mask to find interior, then subtract from original to get boundary
        if boundary_width == 1:
            # Simple boundary: pixels that are 1 but have at least one 0 neighbor
            eroded = binary_erosion(class_mask, structure=np.ones((3, 3)))
            class_boundary = class_mask & (~eroded)
        else:
            # For wider boundaries, use multiple erosions
            eroded = binary_erosion(class_mask, structure=np.ones((3, 3)), iterations=boundary_width)
            # Create a dilated version to get boundary region
            dilated = binary_dilation(eroded, structure=np.ones((3, 3)), iterations=boundary_width)
            class_boundary = dilated & (~eroded)
        
        boundary_mask = boundary_mask | class_boundary
    
    return boundary_mask


def calculate_boundary_iou(pred_mask, gt_mask, num_classes, boundary_width=1, ignore_index=None):
    """
    Calculate Boundary Intersection over Union (Boundary IoU)
    
    Boundary IoU evaluates segmentation quality specifically at object boundaries.
    It extracts boundary regions from both prediction and ground truth, then calculates
    IoU only on these boundary pixels. This is important for applications like
    autonomous driving where edge accuracy (e.g., curbs, obstacles) is critical.
    
    Args:
        pred_mask: Predicted mask, shape [H, W] or [B, H, W], class indices
        gt_mask: Ground truth mask, shape [H, W] or [B, H, W], class indices
        num_classes: Number of classes
        boundary_width: Width of boundary region in pixels (default: 1)
        ignore_index: Class index to ignore (e.g., void/background)
    
    Returns:
        boundary_iou: Mean Boundary IoU across all classes
        boundary_iou_per_class: Boundary IoU for each class
    """
    # Convert to numpy if torch tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Handle batch dimension
    if pred_mask.ndim == 3:
        # Process each sample in batch and average
        batch_size = pred_mask.shape[0]
        boundary_ious = []
        boundary_iou_per_class_all = []
        
        for i in range(batch_size):
            biou, biou_per_class = calculate_boundary_iou(
                pred_mask[i], gt_mask[i], num_classes, boundary_width, ignore_index
            )
            boundary_ious.append(biou)
            boundary_iou_per_class_all.append(biou_per_class)
        
        # Average across batch
        boundary_iou = np.mean(boundary_ious)
        boundary_iou_per_class = np.mean(boundary_iou_per_class_all, axis=0)
        return boundary_iou, boundary_iou_per_class
    
    # Extract boundaries
    pred_boundary = extract_boundary(pred_mask, boundary_width)
    gt_boundary = extract_boundary(gt_mask, boundary_width)
    
    # Create boundary region mask (union of pred and gt boundaries)
    boundary_region = pred_boundary | gt_boundary
    
    if np.sum(boundary_region) == 0:
        # No boundaries found, return NaN
        boundary_iou_per_class = np.full(num_classes, np.nan, dtype=np.float64)
        return np.nan, boundary_iou_per_class
    
    # Mask both predictions and ground truth to boundary region only
    pred_boundary_masked = pred_mask.copy()
    gt_boundary_masked = gt_mask.copy()
    pred_boundary_masked[~boundary_region] = ignore_index if ignore_index is not None else -1
    gt_boundary_masked[~boundary_region] = ignore_index if ignore_index is not None else -1
    
    # Calculate IoU on boundary region only
    # Use a modified ignore_index to exclude non-boundary pixels
    boundary_iou_per_class, _, _ = calculate_iou(
        pred_boundary_masked, gt_boundary_masked, num_classes, 
        ignore_index=ignore_index if ignore_index is not None else -1
    )
    
    # Calculate mean Boundary IoU
    valid_ious = boundary_iou_per_class[~np.isnan(boundary_iou_per_class)]
    if len(valid_ious) > 0:
        boundary_iou = np.mean(valid_ious)
    else:
        boundary_iou = 0.0
    
    return boundary_iou, boundary_iou_per_class


def calculate_boundary_iou_batch(pred_masks, gt_masks, num_classes, boundary_width=1, ignore_index=None):
    """
    Calculate Boundary IoU for a batch of predictions
    
    Args:
        pred_masks: Predicted masks, shape [B, H, W], class indices
        gt_masks: Ground truth masks, shape [B, H, W], class indices
        num_classes: Number of classes
        boundary_width: Width of boundary region in pixels (default: 1)
        ignore_index: Class index to ignore
    
    Returns:
        boundary_iou: Mean Boundary IoU across all classes and all samples
        boundary_iou_per_class: Average Boundary IoU for each class across all samples
    """
    batch_size = pred_masks.shape[0]
    
    # Accumulate boundary intersection and union across batch
    total_boundary_intersection = np.zeros(num_classes, dtype=np.float64)
    total_boundary_union = np.zeros(num_classes, dtype=np.float64)
    
    for i in range(batch_size):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i]
        
        # Convert to numpy if torch tensors
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # Extract boundaries
        pred_boundary = extract_boundary(pred_mask, boundary_width)
        gt_boundary = extract_boundary(gt_mask, boundary_width)
        
        # Create boundary region mask
        boundary_region = pred_boundary | gt_boundary
        
        if np.sum(boundary_region) == 0:
            continue
        
        # Mask to boundary region only
        pred_boundary_masked = pred_mask.copy()
        gt_boundary_masked = gt_mask.copy()
        pred_boundary_masked[~boundary_region] = ignore_index if ignore_index is not None else -1
        gt_boundary_masked[~boundary_region] = ignore_index if ignore_index is not None else -1
        
        # Flatten
        pred_flat = pred_boundary_masked.flatten()
        gt_flat = gt_boundary_masked.flatten()
        
        # Ignore non-boundary pixels
        ignore_val = ignore_index if ignore_index is not None else -1
        valid_mask = (gt_flat != ignore_val)
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]
        
        # Calculate intersection and union for each class on boundaries
        for cls in range(num_classes):
            pred_cls = (pred_flat == cls)
            gt_cls = (gt_flat == cls)
            
            total_boundary_intersection[cls] += np.logical_and(pred_cls, gt_cls).sum()
            total_boundary_union[cls] += np.logical_or(pred_cls, gt_cls).sum()
    
    # Calculate Boundary IoU for each class
    boundary_iou_per_class = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        if total_boundary_union[cls] > 0:
            boundary_iou_per_class[cls] = total_boundary_intersection[cls] / total_boundary_union[cls]
        else:
            boundary_iou_per_class[cls] = np.nan
    
    # Calculate mean Boundary IoU
    valid_ious = boundary_iou_per_class[~np.isnan(boundary_iou_per_class)]
    if len(valid_ious) > 0:
        boundary_iou = np.mean(valid_ious)
    else:
        boundary_iou = 0.0
    
    return boundary_iou, boundary_iou_per_class
