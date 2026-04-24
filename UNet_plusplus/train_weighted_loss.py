import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.visualizer import plot_loss_curve, visualize_prediction, plot_metric_curve, plot_multi_curve

# For evaluation metrics, we will implement them in a separate module (e.g., evaluation_matrix.py) and import here.
from evaluation_matrix.miou import calculate_miou, calculate_iou
from evaluation_matrix.pixel_accuracy import calculate_pixel_accuracy
from evaluation_matrix.dice_coefficient import calculate_mean_dice
from evaluation_matrix.fwiou import calculate_fwiou_batch


# =============================================================================
# [Enhancement 1 & 3] Loss Functions: Dice + Focal + CrossEntropy (TripleLoss)
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=None, smooth=1e-6):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)  # [B, C, H, W]
        
        # One-hot encode target: [B, H, W] -> [B, C, H, W]
        target_one_hot = F.one_hot(
            target.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Zero out the ignore index channel
        if self.ignore_index is not None:
            target_one_hot[:, self.ignore_index] = 0
            pred = pred.clone()
            pred[:, self.ignore_index] = 0

        # Vectorised intersection and union across all classes at once
        dims = (0, 2, 3)  # sum over batch, H, W — keep class dim
        intersection = (pred * target_one_hot).sum(dims)       # [C]
        union        = pred.sum(dims) + target_one_hot.sum(dims)  # [C]

        # Only average over classes that are present
        valid = union > 0
        dice  = torch.where(
            valid,
            1 - (2 * intersection + self.smooth) / (union + self.smooth),
            torch.zeros_like(intersection),
        )
        return dice[valid].mean() if valid.any() else pred.sum() * 0.0

class FocalLoss(nn.Module):
    """
    Focal Loss down-weights easy (well-classified) examples so the model focuses
    its capacity on hard, misclassified pixels — particularly beneficial for rare
    classes that are systematically easy to ignore.
    gamma=2.0 is the standard value from the original RetinaNet paper.
    """
    def __init__(self, gamma=2.0, ignore_index=None):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ignore = self.ignore_index if self.ignore_index is not None else -100
        ce     = F.cross_entropy(pred, target, reduction='none', ignore_index=ignore)
        pt     = torch.exp(-ce)
        focal  = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class TripleLoss(nn.Module):
    """
    Weighted combination of CrossEntropy + Dice + Focal losses.
      - CrossEntropy (with median-frequency class weights): stable gradient signal
      - Dice: directly optimises IoU-like overlap for imbalanced classes
      - Focal: suppresses easy dominant-class pixels, amplifies rare-class gradients
    Default split: CE=0.4, Dice=0.4, Focal=0.2.
    """
    def __init__(self, num_classes, ignore_index=None, class_weights=None,
                 ce_weight=0.4, dice_weight=0.4, focal_weight=0.2):
        super().__init__()
        # [Enhancement 3] Pass median-frequency class_weights into CrossEntropyLoss
        self.ce    = nn.CrossEntropyLoss(weight=class_weights,  label_smoothing=0.1)
        self.dice  = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=2.0, ignore_index=ignore_index)
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.ce_weight    * self.ce(pred, target)   +
                self.dice_weight  * self.dice(pred, target) +
                self.focal_weight * self.focal(pred, target))


# =============================================================================
# [Enhancement 3] Median Frequency Class Weight Computation
# =============================================================================

def compute_class_weights(class_stats_csv: str, void_index: int,
                           num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Computes per-class loss weights using median frequency balancing:
        weight[c] = median_freq / freq[c]
    Rare classes (low freq) receive a higher weight so that CrossEntropyLoss
    penalises their misclassification more strongly than dominant classes.
    The Void class weight is forced to 0 so it never contributes to the loss.
    """
    stats  = pd.read_csv(class_stats_csv)
    col    = "pixels_train" if "pixels_train" in stats.columns else "total_pixels"
    counts = stats[col].values.astype(np.float64)
    counts[void_index] = 0.0
    
    # Sqrt smoothing: less aggressive than median frequency,
    # prevents extreme weights for near-absent classes
    total  = counts.sum() + 1e-12
    freq   = counts / total
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(freq > 0, 1.0 / np.sqrt(freq), 0.0)
    weights[void_index] = 0.0
    weights = np.clip(weights, 0.0, 3.0)
    # Normalise so mean weight = 1
    valid = weights[weights > 0]
    weights = weights / valid.mean()
    weights[void_index] = 0.0
    
    print(f"Sqrt weights: min={weights[weights>0].min():.2f}, "
          f"max={weights.max():.2f}, "
          f"median={np.median(weights[weights>0]):.2f}")
    print(f"[*] Class weights computed via sqrt smoothing")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# =============================================================================
# [Enhancement 6] Data Augmentation — driving-scene-aware pipeline
# =============================================================================

def get_transforms(img_size=512):
    """
    Defines the data augmentation pipelines for training and validation.
    Augmentations are chosen to simulate realistic variation in dashcam footage:
      - Geometric: flip, affine, grid distortion (lens warp)
      - Photometric: colour jitter, greyscale, fog simulation
      - Regularisation: gaussian blur, coarse dropout
    """
    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-10, 10), p=0.4),
        # Randomly apply one photometric distortion per image for colour robustness
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            A.ToGray(p=1.0),                          # simulate low-light / grey camera
            A.RandomFog(fog_coef_range=(0.1, 0.3)),   # simulate foggy / rainy conditions
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # GridDistortion simulates radial lens distortion common in dashcam optics
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32), p=0.3),
    ])

    # Validation: deterministic resize only — no stochastic operations
    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
    ])

    return train_transform, val_transform


# =============================================================================
# Main Training Function
# =============================================================================

def train():
    """
    Main training loop: data loading, forward/backward passes,
    validation, checkpointing, and curve visualisation.
    """
    # ---------------------------------------------------------
    # 1. Hyperparameters & Configuration Setup
    # ---------------------------------------------------------
    # To resume, set RESUME_CHECKPOINT to the path of best.pth.
    # RESUME_HISTORY_CSV is derived automatically — no need to set manually.
    # RESUME_CHECKPOINT = "checkpoints/unetpp_resnet50_improved_v2_XXXXXXXX_XXXX/best.pth"
    RESUME_CHECKPOINT  = None
    RESUME_HISTORY_CSV = (
        os.path.join(
            "outputs",
            os.path.basename(os.path.dirname(RESUME_CHECKPOINT)),
            "training_history.csv",
        )
        if RESUME_CHECKPOINT is not None else None
    )

    # BACKBONE = "efficientnet-b4"  # Recommended upgrade: stronger than b3
    # BACKBONE = "efficientnet-b3"
    # BACKBONE = "resnet34"
    BACKBONE = "resnet50"

    BATCH_SIZE    = 4
    EPOCHS        = 200
    # LEARNING_RATE = 8e-4   # encoder will use LEARNING_RATE * 0.1 (see Section 3)
    LEARNING_RATE = 1e-3   # encoder will use LEARNING_RATE * 0.1 (see Section 3)
    IMG_SIZE      = 512
    NUM_CLASSES   = 32
    VOID_INDEX    = 30     # Void class (RGB 0,0,0) at index 30 in class_dict.csv

    # Path to class_stats.csv produced by scripts/visualize_class_imbalance.py
    # Required for median-frequency class weight computation
    CLASS_STATS_CSV = "../scripts/outputs/class_stats.csv"

    # [Enhancement 4] Number of linear warm-up epochs before cosine annealing
    WARMUP_EPOCHS = 10

    # MODEL_TYPE = "smp"   # "smp" | 
    MODEL_TYPE = "scratch"
    PATIENCE   = 20      # early stopping patience (epochs without val mIoU improvement)

    DATA_ROOT       = "../data/CamVid"
    TRAIN_IMG_DIR   = os.path.join(DATA_ROOT, "train")
    TRAIN_MASK_DIR  = os.path.join(DATA_ROOT, "train_labels_indexed")
    VAL_IMG_DIR     = os.path.join(DATA_ROOT, "val")
    VAL_MASK_DIR    = os.path.join(DATA_ROOT, "val_labels_indexed")
    CLASS_DICT_PATH = os.path.join(DATA_ROOT, "class_dict.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Reuse existing experiment directory when resuming; create a new one otherwise
    if RESUME_CHECKPOINT is not None:
        RESUME_EXP_NAME = os.path.basename(os.path.dirname(RESUME_CHECKPOINT))
        exp_paths = setup_experiment_directories(RESUME_EXP_NAME, base_dirs=["checkpoints", "outputs"])
    else:
        run_name = (
            generate_run_name(model_name="unetpp", backbone=BACKBONE, extra_tag="improved_v2")
            if MODEL_TYPE == "smp"
            else generate_run_name(model_name="unetpp_scratch", backbone='', extra_tag="improved_v2")
        )
        exp_paths = setup_experiment_directories(run_name, base_dirs=["checkpoints", "outputs"])
    print(f"[*] Experiment initialized: {exp_paths}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    train_tfm, val_tfm = get_transforms(img_size=IMG_SIZE)

    train_dataset = CamVidDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, CLASS_DICT_PATH, transform=train_tfm)
    val_dataset   = CamVidDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   CLASS_DICT_PATH, transform=val_tfm)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    index_to_color = {v: k for k, v in train_dataset.color_to_index.items()}

    # ---------------------------------------------------------
    # 3. Model, Loss, and Optimizer Initialization
    # ---------------------------------------------------------
    model = build_unetplusplus(
        backbone_name=BACKBONE, num_classes=NUM_CLASSES, model_type=MODEL_TYPE
    ).to(device)

    
    # Load checkpoint weights if resuming from a previous run
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"[*] Loaded checkpoint from {RESUME_CHECKPOINT}")
    else:
        print("[*] No checkpoint found, training from scratch.")

    # [Enhancement 3] Compute median-frequency class weights from pixel statistics
    class_weights = None
    if os.path.exists(CLASS_STATS_CSV):
        class_weights = compute_class_weights(CLASS_STATS_CSV, VOID_INDEX, NUM_CLASSES, device)
    else:
        print(f"[!] {CLASS_STATS_CSV} not found — training without class weights.")

    # [Enhancement 1 & 3] TripleLoss: CE (weighted) + Dice + Focal
    criterion = TripleLoss(
        num_classes=NUM_CLASSES,
        ignore_index=VOID_INDEX,
        class_weights=class_weights,
        ce_weight=0.2, dice_weight=0.5, focal_weight=0.3,
    )

    # [Enhancement 5] Differential learning rates:
    # Pretrained encoder uses 10x lower lr to preserve ImageNet features.
    # Decoder, segmentation head, and ASPP train at the full learning rate.
    if MODEL_TYPE == "smp":
        optimizer = optim.AdamW([
            {"params": model.encoder.parameters(),           "lr": LEARNING_RATE * 0.1},
            {"params": model.decoder.parameters(),           "lr": LEARNING_RATE},
            {"params": model.segmentation_head.parameters(), "lr": LEARNING_RATE},
        ], weight_decay=1e-3)
    else:
        optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,
        )

    # [Enhancement 4] Linear warm-up for WARMUP_EPOCHS, then cosine anneal to eta_min=1e-6.
    # Warm-up prevents large gradient updates in early epochs when the decoder is random.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    # ---------------------------------------------------------
    # 4. Core Training Loop
    # ---------------------------------------------------------
    best_val_miou    = 0.0
    patience_counter = 0

    if RESUME_HISTORY_CSV is not None and os.path.exists(RESUME_HISTORY_CSV):
        resume_df          = pd.read_csv(RESUME_HISTORY_CSV)
        history_train_loss = resume_df["train_loss"].tolist()
        history_val_loss   = resume_df["val_loss"].tolist()
        history_val_miou   = resume_df["val_miou"].tolist()
        history_val_acc    = resume_df["val_acc"].tolist()
        history_val_dice   = resume_df["val_dice"].tolist() if "val_dice" in resume_df.columns else []
        history_val_fwiou = resume_df["val_fwiou"].tolist() if "val_fwiou" in resume_df.columns else []
        best_val_miou      = max(history_val_miou)
        START_EPOCH        = len(history_train_loss) + 1
        print(f"[*] Resumed history from epoch {START_EPOCH - 1}, continuing from epoch {START_EPOCH}")
        print(f"[*] Restored best_val_miou: {best_val_miou:.4f}")
    else:
        history_train_loss = []
        history_val_loss   = []
        history_val_miou   = []
        history_val_acc    = []
        history_val_dice   = []
        history_val_fwiou  = []  # FWIoU history for plotting
        START_EPOCH        = 1

    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS):
        print(f"\n--- Epoch {epoch}/{START_EPOCH + EPOCHS - 1} ---")

        # ---- Train Phase ----
        model.train()
        epoch_train_loss = 0.0
        scaler = torch.amp.GradScaler('cuda')

        train_bar = tqdm(train_loader, desc="Training")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # [Enhancement 7] ASPP forward pass:
                # 1. Run encoder to get multi-scale feature maps
                # 2. Fuse them through the UNet++ decoder
                # 3. Apply ASPP for multi-scale context before the final head
                outputs = model(images)
                loss     = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # ---- Validation Phase ----
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc  = 0.0
        epoch_val_dice = 0.0
        epoch_val_fwiou = 0.0

        # Accumulate intersection and union globally for unbiased mIoU
        total_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_union        = np.zeros(NUM_CLASSES, dtype=np.float64)
        # Accumulate per-class pixel counts for FWIoU frequency weight computation
        total_class_pixels = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_valid_pixels = 0

        val_bar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                p_acc,  _, _ = calculate_pixel_accuracy(preds, masks)
                m_dice, _    = calculate_mean_dice(preds, masks, num_classes=NUM_CLASSES)

                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()

                # Accumulate per-sample intersection / union, excluding Void
                for i in range(preds_np.shape[0]):
                    _, inter, uni = calculate_iou(
                        preds_np[i], masks_np[i],
                        num_classes=NUM_CLASSES,
                        ignore_index=VOID_INDEX,
                    )
                    total_intersection += inter
                    total_union        += uni
                    # Accumulate pixel frequency statistics for FWIoU (excluding Void)
                    gt_flat = masks_np[i].flatten()
                    valid   = gt_flat[gt_flat != VOID_INDEX]
                    total_valid_pixels += len(valid)
                    for c in range(NUM_CLASSES):
                        total_class_pixels[c] += np.sum(valid == c)

                epoch_val_acc  += p_acc
                epoch_val_dice += m_dice

        # Globally correct mIoU — absent and Void classes excluded via NaN
        iou_per_class = np.where(
            total_union > 0,
            total_intersection / total_union,
            np.nan,
        )
        iou_per_class[VOID_INDEX] = np.nan
        avg_val_miou = float(np.nanmean(iou_per_class))

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc  = epoch_val_acc  / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        # Compute FWIoU using accumulated pixel frequency weights
        freq_weights = np.where(
            total_valid_pixels > 0,
            total_class_pixels / total_valid_pixels,
            0.0,
        )
        fwiou_valid_mask = ~np.isnan(iou_per_class) & (freq_weights > 0)
        if np.any(fwiou_valid_mask):
            w = freq_weights[fwiou_valid_mask]
            w = w / w.sum()  # Normalise weights to sum to 1
            avg_val_fwiou = float(np.sum(iou_per_class[fwiou_valid_mask] * w))
        else:
            avg_val_fwiou = 0.0

        history_val_loss.append(avg_val_loss)
        history_val_miou.append(avg_val_miou)
        history_val_acc.append(avg_val_acc)
        history_val_dice.append(avg_val_dice)
        history_val_fwiou.append(avg_val_fwiou)
        print(f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f} "
              f"| Val FWIoU: {avg_val_fwiou:.4f} | Val Acc: {avg_val_acc:.4f} | Val Dice: {avg_val_dice:.4f}")

        history_df = pd.DataFrame({
            'train_loss': history_train_loss,
            'val_loss':   history_val_loss,
            'val_miou':   history_val_miou,
            'val_acc':    history_val_acc,
            'val_fwiou':  history_val_fwiou
        })
        history_df.to_csv(
            os.path.join(exp_paths['outputs'], 'training_history.csv'), index=False
        )

        scheduler.step()

        # ---------------------------------------------------------
        # 5. Checkpointing & Visualization
        # ---------------------------------------------------------
        if avg_val_miou > best_val_miou:
            best_val_miou    = avg_val_miou
            patience_counter = 0
            save_path        = os.path.join(exp_paths['checkpoints'], "best.pth")
            # Save both model and ASPP weights so the checkpoint is self-contained
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path} (mIoU: {best_val_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[*] Early stopping triggered after {PATIENCE} epochs without improvement.")
                break
        print(f"[*] Current LR: {scheduler.get_last_lr()}")
        # print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        # print(class_weights)

        plot_metric_curve(history_train_loss, history_val_loss,
                          metric_name="Loss",           save_dir=exp_paths['outputs'])
        
        # Figure 2: IoU metrics (mIoU + FWIoU)
        plot_multi_curve(
            {"Val mIoU": history_val_miou, "Val FWIoU": history_val_fwiou},
            title="UNet++ IoU Metrics Over Epochs",
            save_dir=exp_paths['outputs'],
            filename="iou_curve.png",
        )

        # Figure 3: Accuracy metrics (Pixel Accuracy + Mean Dice)
        plot_multi_curve(
            {"Val Pixel Accuracy": history_val_acc, "Val Mean Dice": history_val_dice},
            title="UNet++ Accuracy Metrics Over Epochs",
            save_dir=exp_paths['outputs'],
            filename="accuracy_curve.png",
        )
        
        # plot_metric_curve(None, history_val_miou,
        #                   metric_name="mIoU",           save_dir=exp_paths['outputs'])
        # plot_metric_curve(None, history_val_acc,
        #                   metric_name="Pixel Accuracy", save_dir=exp_paths['outputs'])
        # plot_metric_curve(None, history_val_dice,
        #                   metric_name="Mean Dice",      save_dir=exp_paths['outputs'])

    print("\nTraining Complete! You can now evaluate the best.pth checkpoint.")


if __name__ == '__main__':
    train()