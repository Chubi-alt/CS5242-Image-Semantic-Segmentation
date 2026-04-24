import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CamVidDataset(Dataset):
    """
    Custom Dataset class for the CamVid dataset.

    This class handles loading images and their corresponding semantic
    segmentation masks. It converts RGB mask values to integer class
    indices based on the provided class dictionary.
    """

    def __init__(self, images_dir, masks_dir, class_dict_path, transform=None):
        """
        Initializes the dataset object.

        Args:
            images_dir (str): Path to the directory containing input images.
            masks_dir (str): Path to the directory containing ground truth masks.
            class_dict_path (str): Path to the CSV file mapping class names to RGB values.
            transform (callable, optional): Optional transform to be applied on a sample 
                                            (e.g., albumentations augmentations).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Get a sorted list of image filenames to ensure proper alignment
        self.image_names = sorted(os.listdir(images_dir))

        # Load class dictionary to map RGB values to class indices
        class_dict = pd.read_csv(class_dict_path)
        
        # Create a dictionary mapping (R, G, B) tuples to integer indices (0, 1, 2...)
        self.color_to_index = {
            (r, g, b): idx for idx, (r, g, b) in enumerate(
                zip(class_dict['r'], class_dict['g'], class_dict['b'])
            )
        }

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.image_names)

    def _rgb_to_index(self, mask):
        """
        Converts an RGB mask image to a 2D array of class indices.

        Args:
            mask (numpy.ndarray): RGB mask image of shape (H, W, 3).

        Returns:
            numpy.ndarray: 2D array of shape (H, W) containing class indices.
        """
        # Initialize an empty array for the index mask with the same spatial dimensions
        index_mask = np.zeros(mask.shape[:2], dtype=np.int64)

        # Iterate through the color dictionary and map colors to corresponding indices
        for color, idx in self.color_to_index.items():
            # Find all pixels that perfectly match the current RGB color
            match = np.all(mask == color, axis=-1)
            index_mask[match] = idx

        return index_mask

    def __getitem__(self, idx):
        """
        Retrieves a single dataset sample (image and mask) by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) where both are torch.Tensor objects.
        """
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # CamVid mask filenames often have an '_L' suffix (e.g., '0001TP_006690.png' -> '0001TP_006690_L.png')
        # Adjust this logic if your specific data split uses exact matching names.
        mask_name = img_name.replace('.png', '_L.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Fallback in case masks have the exact same filename as images
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, img_name)

        # Read image and mask using OpenCV (defaults to BGR), then convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(mask_path)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Convert the RGB mask to a class index mask
        # mask = self._rgb_to_index(mask)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply data augmentations or transformations if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure outputs are PyTorch tensors with correct data types and channel dimensions (C, H, W)
        if not isinstance(image, torch.Tensor):
            # Normalize image to [0, 1] and change shape from (H, W, C) to (C, H, W)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if not isinstance(mask, torch.Tensor):
            # Masks must be of type long (int64) for CrossEntropyLoss
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask