"""
Training script for UNet baseline model
Random initialization, no pretrained weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from unet_model import UNet
from dataset import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate pixel accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()
    
    epoch_loss = running_loss / len(val_loader)
    pixel_acc = correct_pixels / total_pixels
    
    return epoch_loss, pixel_acc


def get_checkpoint_num_classes(checkpoint):
    """Infer the model output classes from a saved checkpoint."""
    state_dict = checkpoint['model_state_dict']
    return state_dict['outc.bias'].shape[0]


def init_wandb(args, num_classes, device):
    """Initialize Weights & Biases logging if requested."""
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError:
        print('Warning: wandb is not installed. Continuing without wandb logging.')
        return None

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                'data_root': args.data_root,
                'class_dict': args.class_dict,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'num_workers': args.num_workers,
                'save_dir': args.save_dir,
                'resume': args.resume,
                'early_stopping_patience': args.early_stopping_patience,
                'num_classes': num_classes,
                'device': str(device),
            },
        )
        print('wandb initialized successfully.')
        return run
    except Exception as exc:
        print(f'Warning: failed to initialize wandb ({exc}). Continuing without wandb logging.')
        return None


def main():
    parser = argparse.ArgumentParser(description='Train UNet Baseline')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory of dataset')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Stop training if validation loss does not improve for this many epochs (default: 10)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='unet-baseline-segmentation',
                        help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity/team name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb run name (optional)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get dataloaders
    print('Loading datasets...')
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        class_dict_path=args.class_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    class_dict = pd.read_csv(args.class_dict)
    num_classes = len(class_dict)

    # If resuming from an older checkpoint, keep the saved output size.
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_num_classes = get_checkpoint_num_classes(resume_checkpoint)
        if checkpoint_num_classes != num_classes:
            print(
                f'Warning: checkpoint has {checkpoint_num_classes} output classes, '
                f'but class_dict.csv defines {num_classes} classes. '
                'Training will resume with the checkpoint class count.'
            )
            num_classes = checkpoint_num_classes

    wandb_run = init_wandb(args, num_classes, device)

    # Create model (random initialization)
    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    model = model.to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Loss function (CrossEntropyLoss for semantic segmentation)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Pixel Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_pixel_acc': val_acc,
                'learning_rate': current_lr,
                'best_val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            })
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss,
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pth'))
            print(f'New best model saved! Val Loss: {val_loss:.4f}')
        else:
            print(
                'Early stopping counter: '
                f'{epochs_without_improvement}/{args.early_stopping_patience}'
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                '\nEarly stopping triggered: '
                f'validation loss did not improve for {args.early_stopping_patience} consecutive epochs.'
            )
            if wandb_run is not None:
                wandb_run.summary['stopped_early'] = True
                wandb_run.summary['early_stopping_reason'] = (
                    f'No validation loss improvement for {args.early_stopping_patience} epochs'
                )
            break
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    if wandb_run is not None:
        wandb_run.summary['best_val_loss'] = best_val_loss
        wandb_run.summary['completed_epochs'] = epoch + 1
        if 'stopped_early' not in wandb_run.summary:
            wandb_run.summary['stopped_early'] = False
        wandb_run.finish()


if __name__ == '__main__':
    main()
