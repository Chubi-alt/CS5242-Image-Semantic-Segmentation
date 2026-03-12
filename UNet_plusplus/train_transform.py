import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
# Add the project root to sys.path to access the sibling evaluation_matrix folder
sys.path.append('..')

# Import custom modules built in previous steps
from utils.dataset import CamVidDataset
from models.builder import build_unetplusplus
from utils.helpers import generate_run_name, setup_experiment_directories
from utils.visualizer import plot_loss_curve, visualize_prediction, plot_metric_curve

# For evaluation metrics, we will implement them in a separate module (e.g., evaluation_matrix.py) and import here.
from evaluation_matrix.miou import calculate_miou, calculate_iou
from evaluation_matrix.pixel_accuracy import calculate_pixel_accuracy
from evaluation_matrix.dice_coefficient import calculate_mean_dice

def get_transforms(img_size=512):
    """
    Defines the data augmentation pipelines for training and validation.
    Data augmentation acts as a regularization technique to prevent overfitting.
    """
    # Training pipeline: Includes random geometric and color transformations
    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                rotate=(-10, 10), p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                    saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32), p=0.3),
    ])

    # Validation pipeline: Strictly deterministic operations (only resize)
    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
    ])

    return train_transform, val_transform

def train():
    """
    The main training loop orchestrating data loading, model forward/backward passes,
    validation, and artifact saving (checkpoints and visualizations).
    """
    # ---------------------------------------------------------
    # 1. Hyperparameters & Configuration Setup
    # ---------------------------------------------------------
    # Resume training from a specific history CSV if needed (set to None to start fresh)
    # RESUME_CHECKPOINT = "checkpoints/unetpp_resnet34_improved_v1_20260311_0315/best.pth"  # Path to a specific checkpoint to resume from (set to None to start fresh)
    # RESUME_HISTORY_CSV = os.path.join(
    #     "outputs",
    #     os.path.basename(os.path.dirname(RESUME_CHECKPOINT)),
    #     "training_history.csv"
    # ) if RESUME_CHECKPOINT is not None else None

    RESUME_CHECKPOINT = None
    RESUME_HISTORY_CSV = None

    BACKBONE = "efficientnet-b3"
    # BACKBONE = "resnet50"
    # BACKBONE = "resnet34"

    # BATCH_SIZE = 16
    BATCH_SIZE = 4
    EPOCHS =  200
    
    LEARNING_RATE = 2e-4
    # LEARNING_RATE = 1e-3
    IMG_SIZE = 512
    NUM_CLASSES = 32
    # Void class index
    VOID_INDEX = 30

    # MODEL_TYPE = "smp" # "smp" for using segmentation_models_pytorch, "scratch" for hand-rolled UNet++
    MODEL_TYPE = "scratch"

    PATIENCE = 15 # For early stopping based on validation mIoU

    # Path configurations based on our previous discussions
    DATA_ROOT = "../data/CamVid"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train_labels_indexed") # Use preprocessed indexed masks
    VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "val_labels_indexed") # Use preprocessed indexed masks
    CLASS_DICT_PATH = os.path.join(DATA_ROOT, "class_dict.csv")

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Generate isolated experiment environment
    if RESUME_CHECKPOINT is not None:
    # Automatically parse the experiment name from the checkpoint path
        RESUME_EXP_NAME = os.path.basename(os.path.dirname(RESUME_CHECKPOINT))
        exp_paths = setup_experiment_directories(RESUME_EXP_NAME, base_dirs=["checkpoints", "outputs"])
    else:
        run_name = generate_run_name(model_name="unetpp", backbone=BACKBONE, extra_tag="improved_v1") if MODEL_TYPE == "smp" else generate_run_name(model_name="unetpp_scratch", backbone='', extra_tag="improved_v1")
        exp_paths = setup_experiment_directories(run_name, base_dirs=["checkpoints", "outputs"])
    print(f"[*] Experiment initialized: {exp_paths}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    train_tfm, val_tfm = get_transforms(img_size=IMG_SIZE)

    train_dataset = CamVidDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, CLASS_DICT_PATH, transform=train_tfm)
    val_dataset = CamVidDataset(VAL_IMG_DIR, VAL_MASK_DIR, CLASS_DICT_PATH, transform=val_tfm)

    # DataLoader handles batching and parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Generate the reverse mapping dict for visualization
    index_to_color = {v: k for k, v in train_dataset.color_to_index.items()}

    # Select a fixed validation image for consistent visual tracking across epochs
    # fixed_val_idx = 0 # You can change this index to pick a specific interesting image
    # fixed_val_image, fixed_val_mask_gt = val_dataset[fixed_val_idx]
    # # Add batch dimension and move to device for inference
    # fixed_val_image_tensor = fixed_val_image.unsqueeze(0).to(device) 

    # ---------------------------------------------------------
    # 3. Model, Loss, and Optimizer Initialization
    # ---------------------------------------------------------
    model = build_unetplusplus(backbone_name=BACKBONE, num_classes=NUM_CLASSES, model_type=MODEL_TYPE).to(device)

    # Load checkpoint weights if resuming from a previous run
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=device))
        print(f"[*] Resumed model weights from: {RESUME_CHECKPOINT}")
    else:
        print("[*] No checkpoint found, training from scratch.")

    # CrossEntropyLoss expects logits (raw unnormalized scores) and integer class indices
    criterion = nn.CrossEntropyLoss()
    # AdamW is generally preferred over Adam for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------------------------------------------------
    # 4. Core Training Loop
    # ---------------------------------------------------------
    best_val_miou = 0.0  # Track best mIoU instead of best loss for model saving
    patience_counter = 0  # Initialise patience counter for early stopping
    if RESUME_HISTORY_CSV is not None and os.path.exists(RESUME_HISTORY_CSV):
        resume_df = pd.read_csv(RESUME_HISTORY_CSV)
        history_train_loss = resume_df["train_loss"].tolist()
        history_val_loss   = resume_df["val_loss"].tolist()
        history_val_miou   = resume_df["val_miou"].tolist()
        history_val_acc    = resume_df["val_acc"].tolist()
        history_val_dice   = resume_df["val_dice"].tolist() if "val_dice" in resume_df.columns else []
        # Restore best_val_miou so early stopping and checkpointing thresholds are consistent
        best_val_miou = max(history_val_miou)
        START_EPOCH = len(history_train_loss) + 1
        print(f"[*] Resumed history from epoch {START_EPOCH - 1}, continuing from epoch {START_EPOCH}")
        print(f"[*] Restored best_val_miou: {best_val_miou:.4f}")
    else:
        history_train_loss = []
        history_val_loss   = []
        history_val_miou   = []
        history_val_acc    = []
        history_val_dice   = []
        START_EPOCH = 1

    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS):
        print(f"\n--- Epoch {epoch}/{START_EPOCH + EPOCHS - 1} ---")
        
        # --- Train Phase ---
        model.train() # Set model to training mode (enables dropout, batchnorm updates)
        epoch_train_loss = 0.0
        
        # Initialize the Scaler at the beginning of train()
        # scaler = torch.cuda.amp.GradScaler()
        scaler = torch.amp.GradScaler('cuda')

        # tqdm provides a nice progress bar
        train_bar = tqdm(train_loader, desc="Training")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            # Zero gradients to prevent accumulation from previous iterations
            optimizer.zero_grad()
            
            # Forward pass
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()

            epoch_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (freezes dropout, uses batchnorm stats)
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        epoch_val_dice = 0.0

        # Accumulate intersection and union across the entire epoch for a globally correct mIoU,
        # instead of averaging per-batch mIoU values which introduces sampling bias.
        total_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
        
        val_bar = tqdm(val_loader, desc="Validation")
        # Disable gradient calculation to save memory and speed up computation
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # 1. Loss calculation
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                # 2. Get predictions for metrics
                # Shape: [Batch, H, W]
                preds = torch.argmax(outputs, dim=1)
                
                # 3. Call teammate's metric functions
                # Note: Move to CPU/Numpy if their scripts require it
                # calculate_pixel_accuracy  (pixel_acc, correct_pixels, total_pixels)
                p_acc, _, _ = calculate_pixel_accuracy(preds, masks)
                # calculate_mean_dice  (mean_dice, dice_per_class)
                m_dice, _ = calculate_mean_dice(preds, masks, num_classes=NUM_CLASSES)

                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()

                # Accumulate per-sample intersection and union, excluding the Void class
                for i in range(preds_np.shape[0]):
                    _, inter, uni = calculate_iou(
                        preds_np[i], masks_np[i],
                        num_classes=NUM_CLASSES,
                        ignore_index=VOID_INDEX
                    )
                    total_intersection += inter
                    total_union += uni

                epoch_val_acc += p_acc
                epoch_val_dice += m_dice

        # Compute globally correct mIoU from accumulated intersection/union across all batches
        iou_per_class = np.where(
            total_union > 0,
            total_intersection / total_union,
            np.nan  # Mark absent classes as NaN so they are excluded from the mean
        )
        # Exclude NaN (absent classes) and the Void class from the final mean
        iou_per_class[VOID_INDEX] = np.nan
        avg_val_miou = float(np.nanmean(iou_per_class))

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        
        history_val_loss.append(avg_val_loss)
        history_val_miou.append(avg_val_miou)
        history_val_acc.append(avg_val_acc)
        history_val_dice.append(avg_val_dice)
        print(f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f} | Val Acc: {avg_val_acc:.4f} | Val Dice: {avg_val_dice:.4f}")

        # save the training history to a DataFrame for easier analysis and visualization later
        history_df = pd.DataFrame({
            'train_loss': history_train_loss,
            'val_loss': history_val_loss,
            'val_miou': history_val_miou,
            'val_acc': history_val_acc
        })
        # save the history DataFrame to a CSV file in the outputs directory for record-keeping and potential future analysis
        history_df.to_csv(os.path.join(exp_paths['outputs'], 'training_history.csv'), index=False)
        
        scheduler.step() # Update learning rate based on the scheduler
        # ---------------------------------------------------------
        # 5. Checkpointing & Visualization
        # ---------------------------------------------------------
        # Save the model if it achieves a new best validation mIoU (better reflects segmentation quality than loss)
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            patience_counter = 0  # Reset patience counter on improvement
            save_path = os.path.join(exp_paths['checkpoints'], "best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path} (mIoU: {best_val_miou:.4f})")
        else:
            patience_counter += 1  # Increment patience counter if no improvement
            if patience_counter >= PATIENCE:
                print(f"[*] Early stopping triggered after {PATIENCE} epochs without improvement.")
                break  # Early stopping if patience is exceeded

        # Update and save the loss curve chart
        # plot_loss_curve(history_train_loss, history_val_loss, 
        #                 save_dir=exp_paths['outputs'], filename="loss_curve.png")
        # 1. Loss
        plot_metric_curve(history_train_loss, history_val_loss, metric_name="Loss", 
                          save_dir=exp_paths['outputs'])
        # 2. mIoU
        # Since we don't calculate mIoU for train (to save time), pass the same list or dummy
        plot_metric_curve(None, history_val_miou, metric_name="mIoU",
                  save_dir=exp_paths['outputs'])
        # 3. Accuracy
        plot_metric_curve(None, history_val_acc, metric_name="Pixel Accuracy",
                  save_dir=exp_paths['outputs'])
        # 4. Dice Coefficient
        plot_metric_curve(None, history_val_dice, metric_name="Mean Dice", 
                          save_dir=exp_paths['outputs'])

    print("\nTraining Complete! You can now evaluate the best.pth checkpoint.")

if __name__ == '__main__':
    train()