"""
Dataset loader for semantic segmentation
Handles RGB images and RGB-encoded label masks
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with RGB-encoded labels"""
    
    def __init__(self, images_dir, labels_dir, class_dict_path, transform=None):
        """
        Args:
            images_dir: path to directory containing input images
            labels_dir: path to directory containing label images
            class_dict_path: path to class_dict.csv file
            transform: optional transform to be applied on a sample
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Load class dictionary
        self.class_dict = pd.read_csv(class_dict_path)
        self.num_classes = len(self.class_dict)
        
        # Create RGB to class index mapping
        self.rgb_to_class = {}
        for idx, row in self.class_dict.iterrows():
            rgb = (int(row['r']), int(row['g']), int(row['b']))
            self.rgb_to_class[rgb] = idx
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # Label files have _L suffix before .png extension
        label_name = img_name.replace('.png', '_L.png')
        label_path = os.path.join(self.labels_dir, label_name)
        
        # Load image and label
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        
        # Convert to numpy arrays
        image = np.array(image)
        label = np.array(label)
        
        # Convert RGB label to class indices
        label_mask = self.rgb_to_class_mask(label)
        
        # Apply transforms if any
        if self.transform:
            # Transform should handle both image and mask
            transformed = self.transform(image=image, mask=label_mask)
            image = transformed['image']
            label_mask = transformed['mask']
            
            # ToTensorV2 converts image to tensor, but mask might still be numpy
            if isinstance(label_mask, np.ndarray):
                label_mask = torch.from_numpy(label_mask).long()
            elif isinstance(label_mask, torch.Tensor):
                # Ensure mask is long type (int64) for CrossEntropyLoss
                label_mask = label_mask.long()
            else:
                label_mask = torch.tensor(label_mask, dtype=torch.long)
        else:
            # Convert to tensors if no transform
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if not isinstance(label_mask, torch.Tensor):
                label_mask = torch.from_numpy(label_mask).long()
            else:
                # Ensure mask is long type (int64) for CrossEntropyLoss
                label_mask = label_mask.long()
        
        return image, label_mask
    
    def rgb_to_class_mask(self, label_rgb):
        """Convert RGB label image to class index mask"""
        h, w = label_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        
        for rgb, class_idx in self.rgb_to_class.items():
            # Find pixels matching this RGB value
            matches = np.all(label_rgb == np.array(rgb), axis=2)
            mask[matches] = class_idx
        
        return mask


def get_dataloaders(data_root, class_dict_path, batch_size=4, num_workers=4, img_size=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: root directory containing train, val, test folders
        class_dict_path: path to class_dict.csv
        batch_size: batch size for dataloaders
        num_workers: number of workers for dataloaders
        img_size: optional tuple (height, width) to resize images. If None, keeps original size.
    Returns:
        train_loader, val_loader, test_loader
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Build transform pipelines
    train_transforms = []
    val_transforms = []
    
    # Add resize if img_size is provided
    if img_size is not None:
        if isinstance(img_size, (int, float)):
            img_size = (int(img_size), int(img_size))
        train_transforms.append(A.Resize(height=img_size[0], width=img_size[1]))
        val_transforms.append(A.Resize(height=img_size[0], width=img_size[1]))
    
    # Add augmentation for training
    train_transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation/test transforms (no augmentation)
    val_transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Define transforms
    train_transform = A.Compose(train_transforms, additional_targets={'mask': 'mask'})
    val_transform = A.Compose(val_transforms, additional_targets={'mask': 'mask'})
    
    # Create datasets
    train_dataset = SegmentationDataset(
        images_dir=os.path.join(data_root, 'train'),
        labels_dir=os.path.join(data_root, 'train_labels'),
        class_dict_path=class_dict_path,
        transform=train_transform
    )
    
    val_dataset = SegmentationDataset(
        images_dir=os.path.join(data_root, 'val'),
        labels_dir=os.path.join(data_root, 'val_labels'),
        class_dict_path=class_dict_path,
        transform=val_transform
    )
    
    test_dataset = SegmentationDataset(
        images_dir=os.path.join(data_root, 'test'),
        labels_dir=os.path.join(data_root, 'test_labels'),
        class_dict_path=class_dict_path,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
