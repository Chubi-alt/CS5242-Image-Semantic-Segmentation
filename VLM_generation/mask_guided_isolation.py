"""
Mask-Guided Instance Isolation
Extract instances by multiplying predicted masks with raw images,
applying padding to filter out background noise.
"""

import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional


def resize_mask_to_image(mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a class-index mask to the target image size using nearest interpolation."""
    target_h, target_w = image_shape
    if mask.shape == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def extract_instances_by_class(
    image: np.ndarray,
    mask: np.ndarray,
    class_idx: int,
    padding: int = 10,
    min_area: int = 100
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Extract instances of a specific class from the image using the mask.
    
    Args:
        image: Original RGB image, shape [H, W, 3]
        mask: Class index mask, shape [H, W]
        class_idx: Class index to extract
        padding: Padding around bounding box to include context
        min_area: Minimum area (in pixels) to consider as valid instance
    
    Returns:
        List of (isolated_instance, bbox) tuples where:
        - isolated_instance: RGB image with background masked out, shape [H', W', 3]
        - bbox: (x_min, y_min, x_max, y_max) bounding box coordinates
    """
    # Create binary mask for the specific class
    class_mask = (mask == class_idx).astype(np.uint8)
    
    # Find connected components (instances)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        class_mask, connectivity=8
    )
    
    instances = []
    
    for label_id in range(1, num_labels):  # Skip background (label 0)
        # Get bounding box
        raw_x_min = stats[label_id, cv2.CC_STAT_LEFT]
        raw_y_min = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # Filter by minimum area
        if area < min_area:
            continue
        
        # Add padding
        max_h = min(image.shape[0], mask.shape[0])
        max_w = min(image.shape[1], mask.shape[1])
        x_min = max(0, raw_x_min - padding)
        y_min = max(0, raw_y_min - padding)
        x_max = min(max_w, raw_x_min + width + padding)
        y_max = min(max_h, raw_y_min + height + padding)
        
        # Extract region
        region_image = image[y_min:y_max, x_min:x_max].copy()
        region_mask = class_mask[y_min:y_max, x_min:x_max]
        
        # Create isolated instance (mask out background)
        isolated_instance = region_image.copy()
        isolated_instance[region_mask == 0] = [0, 0, 0]  # Set background to black
        
        instances.append((isolated_instance, (x_min, y_min, x_max, y_max)))
    
    return instances


def extract_all_instances(
    image: np.ndarray,
    mask: np.ndarray,
    class_indices: Optional[List[int]] = None,
    padding: int = 10,
    min_area: int = 100,
    exclude_classes: Optional[List[int]] = None
) -> dict:
    """
    Extract all instances from the image for specified classes.
    
    Args:
        image: Original RGB image, shape [H, W, 3]
        mask: Class index mask, shape [H, W]
        class_indices: List of class indices to extract (None = all classes)
        padding: Padding around bounding box
        min_area: Minimum area for valid instance
        exclude_classes: List of class indices to exclude (e.g., background/void)
    
    Returns:
        Dictionary mapping class_idx -> list of (isolated_instance, bbox) tuples
    """
    if image.shape[:2] != mask.shape:
        mask = resize_mask_to_image(mask, image.shape[:2])

    if exclude_classes is None:
        exclude_classes = []
    
    # Get unique classes in mask
    unique_classes = np.unique(mask)
    
    # Filter classes
    if class_indices is not None:
        unique_classes = [c for c in unique_classes if c in class_indices]
    
    unique_classes = [c for c in unique_classes if c not in exclude_classes]
    
    all_instances = {}
    
    for class_idx in unique_classes:
        instances = extract_instances_by_class(
            image, mask, class_idx, padding, min_area
        )
        if instances:
            all_instances[class_idx] = instances
    
    return all_instances


def apply_padding_filter(
    isolated_instance: np.ndarray,
    mask_region: np.ndarray,
    padding_mode: str = 'zero',
    blur_sigma: float = 1.0
) -> np.ndarray:
    """
    Apply padding and filtering to reduce background noise.
    
    Args:
        isolated_instance: Isolated instance image, shape [H, W, 3]
        mask_region: Binary mask for the instance region, shape [H, W]
        padding_mode: 'zero' or 'blur' for background handling
        blur_sigma: Sigma for Gaussian blur if padding_mode is 'blur'
    
    Returns:
        Filtered isolated instance
    """
    filtered = isolated_instance.copy()
    
    if padding_mode == 'blur':
        # Apply Gaussian blur to background
        blurred = cv2.GaussianBlur(
            isolated_instance, 
            (0, 0), 
            blur_sigma
        )
        filtered[mask_region == 0] = blurred[mask_region == 0]
    elif padding_mode == 'zero':
        # Already set to zero in extract_instances_by_class
        pass
    
    return filtered


def save_isolated_instances(
    instances: dict,
    output_dir: str,
    base_filename: str,
    class_dict: dict
):
    """
    Save isolated instances to files.
    
    Args:
        instances: Dictionary mapping class_idx -> list of (instance, bbox) tuples
        output_dir: Output directory
        class_dict: Dictionary mapping class_idx -> class_name
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for class_idx, instance_list in instances.items():
        class_name = class_dict.get(class_idx, f'class_{class_idx}')
        
        for inst_idx, (instance, bbox) in enumerate(instance_list):
            filename = f"{base_filename}_class{class_idx}_{class_name}_inst{inst_idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            Image.fromarray(instance).save(filepath)


def load_mask_from_rgb(
    mask_path: str,
    class_dict_path: str
) -> Tuple[np.ndarray, dict]:
    """
    Load RGB-encoded mask and convert to class index mask.
    
    Args:
        mask_path: Path to RGB mask image
        class_dict_path: Path to class_dict.csv
    
    Returns:
        (class_mask, class_dict) where:
        - class_mask: Class index mask, shape [H, W]
        - class_dict: Dictionary mapping class_idx -> class_name
    """
    import pandas as pd
    
    # Load mask
    mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
    
    # Load class dictionary
    class_df = pd.read_csv(class_dict_path)
    
    # Create RGB to class index mapping
    rgb_to_class = {}
    class_dict = {}
    for idx, row in class_df.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        rgb_to_class[rgb] = idx
        class_dict[idx] = row['name']
    
    # Convert RGB mask to class indices
    h, w = mask_rgb.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    
    for rgb, class_idx in rgb_to_class.items():
        matches = np.all(mask_rgb == np.array(rgb), axis=2)
        class_mask[matches] = class_idx
    
    return class_mask, class_dict
