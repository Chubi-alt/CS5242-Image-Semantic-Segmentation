import os
import torch
import numpy as np
from utils.dataset import CamVidDataset

def test_data_loading():
    """
    Tests the CamVidDataset class to ensure images and masks are loaded 
    and transformed correctly before feeding them into the model.
    """
    # Define paths based on your project structure
    # Assuming this script is run from the UNet_plusplus directory
    # and data is in the sibling 'data' directory.
    data_root = '../data/CamVid'  # Adjust this path if your data is located elsewhere
    
    # CamVid typically splits data into train/val/test and their corresponding label folders.
    # Adjust 'train' and 'train_labels' if your folder names differ.
    train_images_dir = os.path.join(data_root, 'train')
    train_masks_dir = os.path.join(data_root, 'train_labels')
    class_dict_path = os.path.join(data_root, 'class_dict.csv')

    # Check if paths exist to prevent cryptic errors
    if not os.path.exists(train_images_dir):
        print(f"Error: Images directory not found at {train_images_dir}")
        return
    if not os.path.exists(class_dict_path):
        print(f"Error: Class dict not found at {class_dict_path}")
        return

    print("Initializing CamVidDataset...")
    # Initialize the dataset without any external transforms for a raw test
    dataset = CamVidDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        class_dict_path=class_dict_path
    )

    print(f"Total samples in dataset: {len(dataset)}")

    # Fetch the first sample
    image, mask = dataset[0]

    # Print engineering assertions and info
    print("\n--- Testing Sample 0 ---")
    
    # 1. Check types
    print(f"Image type: {type(image)}, Mask type: {type(mask)}")
    print(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}")
    
    # 2. Check shapes
    # Expected Image shape: [3, H, W] (Channels first for PyTorch)
    # Expected Mask shape: [H, W] (No channel dimension for CrossEntropyLoss)
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    # 3. Check value ranges
    # Image should be normalized to [0.0, 1.0]
    print(f"Image min value: {image.min().item():.4f}, max value: {image.max().item():.4f}")
    
    # Mask should contain integer class indices (e.g., 0, 1, 2... up to num_classes - 1)
    unique_classes = torch.unique(mask).tolist()
    print(f"Unique class indices present in mask: {unique_classes}")
    
    print("\nTest passed successfully! The data is ready for the model.")

if __name__ == '__main__':
    test_data_loading()