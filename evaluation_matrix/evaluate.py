"""
Main evaluation script
Calculate mIoU, Dice Coefficient, and Pixel Accuracy on test set
"""

import torch
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from miou import calculate_miou_batch
from dice_coefficient import calculate_dice_batch
from pixel_accuracy import calculate_pixel_accuracy_batch


def get_checkpoint_num_classes(checkpoint):
    """Infer the model output classes from a saved checkpoint."""
    state_dict = checkpoint['model_state_dict']
    return state_dict['outc.bias'].shape[0]


def get_class_names(class_dict_path, num_classes):
    """Return class names aligned to the model output size."""
    class_names = []
    if class_dict_path and os.path.exists(class_dict_path):
        class_dict = pd.read_csv(class_dict_path)
        class_names.extend(class_dict['name'].tolist())
    while len(class_names) < num_classes:
        class_names.append(f'extra_class_{len(class_names)}')
    return class_names


def evaluate_model(model, test_loader, device, num_classes, ignore_index=None):
    """
    Evaluate model on test set and calculate all metrics
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to run on
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        metrics: Dictionary containing all metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print('Running inference on test set...')
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)  # [N, H, W]
    all_labels = torch.cat(all_labels, dim=0)  # [N, H, W]
    
    print('Calculating metrics...')
    
    # Calculate mIoU
    miou, iou_per_class = calculate_miou_batch(
        all_preds.numpy(), all_labels.numpy(), num_classes, ignore_index
    )
    
    # Calculate Dice Coefficient
    mean_dice, dice_per_class = calculate_dice_batch(
        all_preds.numpy(), all_labels.numpy(), num_classes, ignore_index
    )
    
    # Calculate Pixel Accuracy
    pixel_acc, correct_pixels, total_pixels = calculate_pixel_accuracy_batch(
        all_preds.numpy(), all_labels.numpy(), ignore_index
    )
    
    metrics = {
        'mIoU': miou,
        'Mean Dice': mean_dice,
        'Pixel Accuracy': pixel_acc,
        'IoU per class': iou_per_class,
        'Dice per class': dice_per_class,
        'Correct pixels': correct_pixels,
        'Total pixels': total_pixels
    }
    
    return metrics


def print_metrics(metrics, class_dict_path=None):
    """Print evaluation metrics"""
    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)
    print(f'Mean IoU (mIoU):        {metrics["mIoU"]:.4f}')
    print(f'Mean Dice Coefficient:   {metrics["Mean Dice"]:.4f}')
    print(f'Pixel Accuracy:          {metrics["Pixel Accuracy"]:.4f}')
    print(f'Correct Pixels:          {metrics["Correct pixels"]:,}')
    print(f'Total Pixels:            {metrics["Total pixels"]:,}')
    
    class_names = get_class_names(class_dict_path, len(metrics['IoU per class']))
    print('\nPer-class IoU:')
    print('-'*60)
    for idx, class_name in enumerate(class_names):
        iou = metrics['IoU per class'][idx]
        if not np.isnan(iou):
            print(f'{class_name:25s}: {iou:.4f}')

    print('\nPer-class Dice Coefficient:')
    print('-'*60)
    for idx, class_name in enumerate(class_names):
        dice = metrics['Dice per class'][idx]
        if not np.isnan(dice):
            print(f'{class_name:25s}: {dice:.4f}')
    
    print('='*60 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet Baseline Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., ../UNet_baseline/checkpoints/best.pth)')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory of dataset')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation (default: 4)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--ignore_index', type=int, default=None,
                        help='Class index to ignore (e.g., void class)')
    
    args = parser.parse_args()
    
    # Import model and dataset (need to add parent directory to path)
    import sys
    unet_baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UNet_baseline'))
    if unet_baseline_path not in sys.path:
        sys.path.insert(0, unet_baseline_path)
    
    from unet_model import UNet
    from dataset import get_dataloaders
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Infer output classes from checkpoint to stay compatible with saved models.
    checkpoint_num_classes = get_checkpoint_num_classes(checkpoint)
    class_dict = pd.read_csv(args.class_dict)
    class_dict_num_classes = len(class_dict)
    num_classes = checkpoint_num_classes
    if checkpoint_num_classes != class_dict_num_classes:
        print(
            f'Warning: checkpoint has {checkpoint_num_classes} output classes, '
            f'but class_dict.csv defines {class_dict_num_classes} classes. '
            'Evaluation will use the checkpoint class count.'
        )
    
    # Create model
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
        img_size=(args.img_size, args.img_size)
    )
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, num_classes, args.ignore_index)
    
    # Print results
    print_metrics(metrics, args.class_dict)
    
    # Save results to a timestamped file under evaluation_matrix/outputs.
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.txt')
    with open(output_file, 'w') as f:
        f.write('Evaluation Results\n')
        f.write('='*60 + '\n')
        f.write(f'Mean IoU (mIoU):        {metrics["mIoU"]:.4f}\n')
        f.write(f'Mean Dice Coefficient:   {metrics["Mean Dice"]:.4f}\n')
        f.write(f'Pixel Accuracy:          {metrics["Pixel Accuracy"]:.4f}\n')
        f.write(f'Correct Pixels:          {metrics["Correct pixels"]:,}\n')
        f.write(f'Total Pixels:            {metrics["Total pixels"]:,}\n')
        
        class_names = get_class_names(args.class_dict, len(metrics['IoU per class']))
        f.write('\nPer-class IoU:\n')
        f.write('-'*60 + '\n')
        for idx, class_name in enumerate(class_names):
            iou = metrics['IoU per class'][idx]
            if not np.isnan(iou):
                f.write(f'{class_name:25s}: {iou:.4f}\n')

        f.write('\nPer-class Dice Coefficient:\n')
        f.write('-'*60 + '\n')
        for idx, class_name in enumerate(class_names):
            dice = metrics['Dice per class'][idx]
            if not np.isnan(dice):
                f.write(f'{class_name:25s}: {dice:.4f}\n')
    
    print(f'Results saved to {output_file}')


if __name__ == '__main__':
    main()
