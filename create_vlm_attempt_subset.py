"""
Create a subset of test images for VLM evaluation
Randomly selects 5 images from each prefix
"""

import os
import random
import shutil
import argparse
from collections import defaultdict


def create_vlm_attempt_subset(test_dir, output_dir, images_per_prefix=5, seed=None):
    """
    Create a subset of test images by randomly selecting images from each prefix.
    
    Args:
        test_dir: Directory containing test images
        output_dir: Directory to save selected images
        images_per_prefix: Number of images to select from each prefix (default: 5)
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    # Group by prefix (first part before underscore)
    prefixes = defaultdict(list)
    for f in image_files:
        prefix = f.split('_')[0]
        prefixes[prefix].append(f)
    
    print(f"Found {len(image_files)} images with {len(prefixes)} prefixes:")
    for prefix in sorted(prefixes.keys()):
        print(f"  {prefix}: {len(prefixes[prefix])} images")
    
    # Select images from each prefix
    selected_images = []
    for prefix in sorted(prefixes.keys()):
        available = prefixes[prefix]
        n_select = min(images_per_prefix, len(available))
        selected = random.sample(available, n_select)
        selected_images.extend(selected)
        
        # Copy selected images
        for img_file in selected:
            src = os.path.join(test_dir, img_file)
            dst = os.path.join(output_dir, img_file)
            shutil.copy2(src, dst)
    
    print(f"\nCreated vlm_attempt subset with {len(selected_images)} images:")
    for prefix in sorted(prefixes.keys()):
        count = len([f for f in selected_images if f.startswith(prefix)])
        print(f"  {prefix}: {count} images")
    
    # List selected images
    print(f"\nSelected images:")
    for img in sorted(selected_images):
        print(f"  {img}")
    
    return selected_images


def main():
    parser = argparse.ArgumentParser(
        description='Create a subset of test images for VLM evaluation'
    )
    parser.add_argument(
        '--test_dir', 
        type=str, 
        default='data/test',
        help='Directory containing test images (default: data/test)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/vlm_attempt',
        help='Output directory for selected images (default: data/vlm_attempt)'
    )
    parser.add_argument(
        '--images_per_prefix', 
        type=int, 
        default=5,
        help='Number of images to select from each prefix (default: 5)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    create_vlm_attempt_subset(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        images_per_prefix=args.images_per_prefix,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
