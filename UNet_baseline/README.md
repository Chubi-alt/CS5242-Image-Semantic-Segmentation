# UNet Baseline for Semantic Segmentation

This directory contains a UNet baseline model implementation for semantic segmentation with random initialization (no pretrained weights).

## Files

- `unet_model.py`: UNet architecture implementation
- `dataset.py`: Dataset loader for RGB images and RGB-encoded label masks
- `train.py`: Training script
- `test.py`: Testing script to generate segmentation results on test set
- `requirements.txt`: Required Python packages

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the UNet baseline model:

```bash
python train.py --data_root ../data --class_dict ../data/class_dict.csv --batch_size 4 --epochs 100 --lr 0.001 --early_stopping_patience 10 --use_wandb --wandb_project unet-baseline-segmentation
```

The training and testing pipelines keep the original image resolution instead of resizing inputs.

### Arguments

- `--data_root`: Root directory containing train, val, test folders (default: `../data`)
- `--class_dict`: Path to class_dict.csv file (default: `../data/class_dict.csv`)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Number of data loading workers (default: 4)
- `--save_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--resume`: Path to checkpoint to resume from (optional)
- `--early_stopping_patience`: Early stopping patience based on validation loss (default: 10)
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: wandb project name (default: `unet-baseline-segmentation`)
- `--wandb_entity`: wandb entity/team name (optional)
- `--wandb_run_name`: wandb run name (optional)

### wandb Logging

To use wandb logging, first log in:

```bash
wandb login
```

Then launch training with wandb enabled:

```bash
python train.py \
    --data_root ../data \
    --class_dict ../data/class_dict.csv \
    --batch_size 4 \
    --epochs 100 \
    --lr 0.001 \
    --early_stopping_patience 10 \
    --use_wandb \
    --wandb_project unet-baseline-segmentation \
    --wandb_run_name exp1
```

The training script logs `train_loss`, `val_loss`, `val_pixel_acc`, learning rate, best validation loss, and early stopping progress to wandb.

### Model Architecture

The UNet model consists of:
- Encoder: 4 downsampling blocks with double convolutions
- Decoder: 4 upsampling blocks with skip connections
- Output: 1x1 convolution to produce 32 class predictions

The model uses random initialization (no pretrained weights).

### Dataset Structure

Expected data structure:
```
data/
├── train/
│   └── *.png
├── train_labels/
│   └── *.png
├── val/
│   └── *.png
├── val_labels/
│   └── *.png
├── test/
│   └── *.png
├── test_labels/
│   └── *.png
└── class_dict.csv
```

Labels are RGB-encoded images where each pixel's RGB value corresponds to a class defined in `class_dict.csv`.
Images and labels are processed at their original dataset resolution (no resize).

### Testing

Generate segmentation results on test set:

```bash
python test.py \
    --checkpoint checkpoints/best.pth \
    --data_root ../data \
    --class_dict ../data/class_dict.csv \
    --output_dir ./test_results \
    --batch_size 4
```

### Arguments for Testing

- `--checkpoint`: Path to checkpoint file (required)
- `--data_root`: Root directory of dataset (default: `../data`)
- `--class_dict`: Path to class_dict.csv (default: `../data/class_dict.csv`)
- `--output_dir`: Directory to save segmentation results (default: `./test_results`)
- `--batch_size`: Batch size for testing (default: 4)
- `--num_workers`: Number of data loading workers (default: 4)

The test script will save RGB-encoded segmentation masks to the output directory.

### Checkpoints

Checkpoints are saved in the `checkpoints` directory:
- `latest.pth`: Latest checkpoint
- `best.pth`: Best checkpoint (lowest validation loss)
