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
from evaluation_matrix.miou import calculate_miou
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
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # ToTensorV2 converts numpy arrays to PyTorch tensors and scales to [0, 1] automatically
        # if the input is uint8. However, our dataset class currently handles the tensor conversion.
        # We will keep it simple here and let the dataset handle the final tensor conversion.
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
    # Define experimental variables for easy modification

    BACKBONE = "efficientnet-b3"
    # BACKBONE = "resnet34"
    # BACKBONE = "resnet50"

    # BATCH_SIZE = 16
    BATCH_SIZE = 4

    EPOCHS =  100

    # LEARNING_RATE = 2e-4
    LEARNING_RATE = 1e-3
    IMG_SIZE = 512
    NUM_CLASSES = 32 # Adjust if your specific class_dict uses a subset (e.g., 11)
    
    # MODEL_TYPE = "smp" # "smp" for using segmentation_models_pytorch, "scratch" for hand-rolled UNet++
    MODEL_TYPE = "scratch" # "smp" for using segmentation_models_pytorch, "scratch" for hand-rolled UNet++

    # Path configurations based on our previous discussions
    DATA_ROOT = "../data/CamVid"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
    # TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train_labels")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train_labels_indexed") # Use preprocessed indexed masks
    VAL_IMG_DIR = os.path.join(DATA_ROOT, "val")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "val_labels_indexed") # Use preprocessed indexed masks
    CLASS_DICT_PATH = os.path.join(DATA_ROOT, "class_dict.csv")

    # Set device to GPU if available (L4 is highly capable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Generate isolated experiment environment
    run_name = generate_run_name(model_name="unetpp", backbone=BACKBONE, extra_tag="baseline") if MODEL_TYPE == "smp" else generate_run_name(model_name="unetpp_scratch", backbone='', extra_tag="baseline")
    exp_paths = setup_experiment_directories(run_name, base_dirs=["checkpoints", "outputs"])
    print(f"[*] Experiment initialized: {run_name}")

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
    
    # CrossEntropyLoss expects logits (raw unnormalized scores) and integer class indices
    criterion = nn.CrossEntropyLoss()
    # AdamW is generally preferred over Adam for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------------------------------------------------
    # 4. Core Training Loop
    # ---------------------------------------------------------
    best_val_loss = float('inf')
    history_train_loss = []
    history_val_loss = []
    history_val_miou = []  # Added for mIoU tracking
    history_val_acc = []   # Added for Accuracy tracking
    history_val_dice = []  # Added for Dice Coefficient tracking

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
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
        epoch_val_miou = 0.0
        epoch_val_acc = 0.0
        epoch_val_dice = 0.0
        
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
                m_iou, _ = calculate_miou(preds, masks, num_classes=NUM_CLASSES)
                # calculate_pixel_accuracy  (pixel_acc, correct_pixels, total_pixels)
                p_acc, _, _ = calculate_pixel_accuracy(preds, masks)
                # calculate_mean_dice  (mean_dice, dice_per_class)
                m_dice, _ = calculate_mean_dice(preds, masks, num_classes=NUM_CLASSES)

                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                epoch_val_miou += m_iou
                epoch_val_acc += p_acc
                epoch_val_dice += m_dice

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_miou = epoch_val_miou / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        
        history_val_loss.append(avg_val_loss)
        history_val_miou.append(avg_val_miou)
        history_val_acc.append(avg_val_acc)
        history_val_dice.append(avg_val_dice)
        print(f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f} | Val Acc: {avg_val_acc:.4f} | Val Dice: {avg_val_dice:.4f}")
        import pandas as pd

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
        # Save the model if it achieves a new best validation loss (our strict evaluation criteria)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(exp_paths['checkpoints'], "best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path} (Loss: {best_val_loss:.4f})")

        # Update and save the loss curve chart
        # plot_loss_curve(history_train_loss, history_val_loss, 
        #                 save_dir=exp_paths['outputs'], filename="loss_curve.png")
        # 1. Loss
        plot_metric_curve(history_train_loss, history_val_loss, metric_name="Loss", 
                          save_dir=exp_paths['outputs'])
        # 2. mIoU
        # Since we don't calculate mIoU for train (to save time), pass the same list or dummy
        plot_metric_curve(history_val_miou, history_val_miou, metric_name="mIoU", 
                          save_dir=exp_paths['outputs'])
        # 3. Accuracy
        plot_metric_curve(history_val_acc, history_val_acc, metric_name="Pixel Accuracy", 
                          save_dir=exp_paths['outputs'])
        # 4. Dice Coefficient
        plot_metric_curve(history_val_dice, history_val_dice, metric_name="Mean Dice", 
                          save_dir=exp_paths['outputs'])

        # with torch.no_grad():
        #     fixed_output = model(fixed_val_image_tensor)
        #     # Apply argmax across the channel dimension to get the predicted class index
        #     fixed_pred_mask = torch.argmax(fixed_output.squeeze(0), dim=0).cpu()
            
        #     visualize_prediction(
        #         image=fixed_val_image, 
        #         mask_gt=fixed_val_mask_gt, 
        #         mask_pred=fixed_pred_mask, 
        #         index_to_color=index_to_color, 
        #         save_dir=exp_paths['outputs'], 
        #         filename=f"pred_epoch_{epoch:03d}.png"
        #     )

    print("\nTraining Complete! You can now evaluate the best.pth checkpoint.")

if __name__ == '__main__':
    train()