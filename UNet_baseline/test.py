"""
Test script for UNet baseline model
Load checkpoint and generate segmentation results on test set
"""

import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd

from unet_model import UNet
from dataset import SegmentationDataset, get_dataloaders


def get_checkpoint_num_classes(checkpoint):
    """Infer the model output classes from a saved checkpoint."""
    state_dict = checkpoint['model_state_dict']
    return state_dict['outc.bias'].shape[0]


def class_to_rgb_mask(class_mask, class_dict):
    """Convert class index mask to RGB mask"""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, row in class_dict.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        matches = (class_mask == idx)
        rgb_mask[matches] = rgb
    
    return rgb_mask


def test_model(model, test_loader, device, output_dir, class_dict_path):
    """Test the model on test set and save segmentation results"""
    model.eval()
    
    # Load class dictionary for RGB conversion
    class_dict = pd.read_csv(class_dict_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            
            # Convert predictions to numpy
            preds_np = preds.cpu().numpy()  # [B, H, W]
            
            # Save each prediction in the batch
            for i in range(preds_np.shape[0]):
                pred_mask = preds_np[i]  # [H, W]
                
                # Convert class indices to RGB
                rgb_mask = class_to_rgb_mask(pred_mask, class_dict)
                
                # Get original image filename
                dataset = test_loader.dataset
                img_idx = batch_idx * test_loader.batch_size + i
                if img_idx < len(dataset):
                    img_name = dataset.image_files[img_idx]
                    output_name = img_name.replace('.png', '_pred.png')
                    output_path = os.path.join(output_dir, output_name)
                    
                    # Save RGB mask
                    Image.fromarray(rgb_mask).save(output_path)
    
    print(f'Segmentation results saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Test UNet Baseline Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoints/best.pth)')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory of dataset')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save segmentation results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model using the checkpoint output size for compatibility.
    num_classes = get_checkpoint_num_classes(checkpoint)
    class_dict = pd.read_csv(args.class_dict)
    if num_classes != len(class_dict):
        print(
            f'Warning: checkpoint has {num_classes} output classes, '
            f'but class_dict.csv defines {len(class_dict)} classes. '
            'Unknown extra classes will be saved as black pixels.'
        )
    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f'Model loaded. Epoch: {checkpoint.get("epoch", "unknown")}, '
          f'Val Loss: {checkpoint.get("val_loss", "unknown"):.4f}')
    
    # Get test dataloader
    print('Loading test dataset...')
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        class_dict_path=args.class_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Test model
    print('Starting testing...')
    test_model(model, test_loader, device, args.output_dir, args.class_dict)
    
    print('Testing completed!')


if __name__ == '__main__':
    main()
