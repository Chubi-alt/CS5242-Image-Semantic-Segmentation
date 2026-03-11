import os
import torch
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom modules
from utils.dataset import CamVidDataset
from models.builder import build_unetplusplus
from utils.visualizer import visualize_prediction

import evaluation_matrix
# Import evaluation metrics from teammate's scripts
from evaluation_matrix.miou import calculate_miou_batch
from evaluation_matrix.pixel_accuracy import calculate_pixel_accuracy_batch
from evaluation_matrix.dice_coefficient import calculate_dice_batch

def evaluate_and_visualize():
    """
    Performs a comprehensive evaluation on the test set and generates 
    colorful qualitative visualizations.
    """
    # ---------------------------------------------------------
    # 1. Configuration & Experiment Setup
    # ---------------------------------------------------------
    # IMPORTANT: Update EXPERIMENT_NAME to match your actual folder in 'checkpoints'
    EXPERIMENT_NAME = "unetpp_efficientnet-b3_improved_v1_20260311_0725"
    # EXPERIMENT_NAME = "unetpp_resnet50_baseline_20260310_0902" 

    # BACKBONE = "resnet34"
    # BACKBONE = "resnet50"
    BACKBONE = "efficientnet-b3"

    NUM_CLASSES = 32
    IMG_SIZE = 512
    BATCH_SIZE = 4 # Higher batch size for faster evaluation on L4 GPU

    # WEATHER = "clear"
    WEATHER = "rainy" # You can switch to "rainy" if you want to evaluate on rainy weather subset
    
    # MODEL_TYPE = "scratch"
    MODEL_TYPE = "smp"

    VOID_INDEX = 30

    # Path setup
    DATA_ROOT = "../data/CamVid"
    # Note: Using indexed masks for numerical calculation
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test") if WEATHER == "clear" else os.path.join(DATA_ROOT, "test_rainy")
    TEST_MASK_DIR = os.path.join(DATA_ROOT, "test_labels_indexed") 
    CLASS_DICT_PATH = os.path.join(DATA_ROOT, "class_dict.csv")
    
    CHECKPOINT_PATH = os.path.join("checkpoints", EXPERIMENT_NAME, "best.pth")
    OUTPUT_DIR = os.path.join("outputs", EXPERIMENT_NAME, "test_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation starting on device: {device}")

    # ---------------------------------------------------------
    # 2. Load Color Map for Visualization
    # ---------------------------------------------------------
    class_dict = pd.read_csv(CLASS_DICT_PATH)
    # Mapping index (0-31) -> RGB tuple (R, G, B) for decoding
    index_to_color = {
        i: (r, g, b) for i, (r, g, b) in enumerate(
            zip(class_dict['r'], class_dict['g'], class_dict['b'])
        )
    }

    # ---------------------------------------------------------
    # 3. Model and Data Loading
    # ---------------------------------------------------------
    print(f"[*] Loading model weights from: {CHECKPOINT_PATH}")
    model = build_unetplusplus(backbone_name=BACKBONE, num_classes=NUM_CLASSES, model_type=MODEL_TYPE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    # Transformation: Resize only, no random augmentations for testing
    test_transform = A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE)])
    test_dataset = CamVidDataset(TEST_IMG_DIR, TEST_MASK_DIR, CLASS_DICT_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ---------------------------------------------------------
    # 4. Quantitative Numerical Evaluation
    # ---------------------------------------------------------
    print(f"[*] Running full inference on test set...")
    
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            
            # Convert logits to class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.append(preds)
            all_masks.append(masks.numpy())

    # Concatenate all batches into large numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Calculate final metrics using teammate's batch functions
    final_miou, iou_per_class = calculate_miou_batch(all_preds, all_masks, num_classes=NUM_CLASSES, ignore_index=VOID_INDEX)
    iou_per_class[VOID_INDEX] = np.nan  # Explicitly exclude Void from per-class display
    final_acc, _, _ = calculate_pixel_accuracy_batch(all_preds, all_masks)
    final_dice, _ = calculate_dice_batch(all_preds, all_masks, num_classes=NUM_CLASSES)

    # Log results to console
    print("\n" + "="*40)
    print(f"FINAL TEST PERFORMANCE ({EXPERIMENT_NAME}) on {WEATHER} test set:")
    print(f"Mean IoU (mIoU): {final_miou:.4f}")
    print(f"Pixel Accuracy:  {final_acc:.4f}")
    print(f"Mean Dice:       {final_dice:.4f}")
    print("="*40)

    # ---------------------------------------------------------
    # 5. Qualitative Qualitative Visualization (Sampled)
    # ---------------------------------------------------------
    print(f"[*] Generating sample visualizations in: {OUTPUT_DIR}")
    # Visualize every 10th image for a broad overview of performance
    for idx in range(0, len(test_dataset), 50):
        img_vis, mask_gt_vis = test_dataset[idx]
        pred_vis = all_preds[idx]
        
        save_name = f"test_prediction_on_{WEATHER}_{idx:03d}.png"
        visualize_prediction(
            image=img_vis,
            mask_gt=mask_gt_vis,
            mask_pred=pred_vis,
            index_to_color=index_to_color,
            save_dir=OUTPUT_DIR,
            filename=save_name
        )
    
    # ---------------------------------------------------------
    # 6. Write Results to Text File
    # ---------------------------------------------------------
    results_file_path = os.path.join(OUTPUT_DIR, f"performance_on_{WEATHER}.txt")
    
    with open(results_file_path, "w") as f:
        f.write("="*40 + "\n")
        f.write(f"TEST PERFORMANCE REPORT\n")
        f.write(f"Experiment: {EXPERIMENT_NAME}\n")
        f.write(f"Backbone:   {BACKBONE}\n")
        f.write(f"Device:     {device}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean IoU (mIoU): {final_miou:.4f}\n")
        f.write(f"Pixel Accuracy:  {final_acc:.4f}\n")
        f.write(f"Mean Dice:       {final_dice:.4f}\n")
        f.write("="*40 + "\n")
        
        # Optional: Save per-class IoU if needed
        f.write("\nPer-Class IoU:\n")
        for cls_idx, iou in enumerate(iou_per_class):
            f.write(f"Class {cls_idx:02d}: {iou:.4f}\n")

    print(f"[*] Performance metrics written to: {results_file_path}")

if __name__ == '__main__':
    evaluate_and_visualize()