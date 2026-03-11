# Evaluation Metrics for Semantic Segmentation

This directory contains evaluation metrics for semantic segmentation tasks.

## Metrics

### 1. Mean Intersection over Union (mIoU)
- **File**: `miou.py`
- **Description**: Standardized Mean Intersection over Union calculates the average IoU across all classes
- **Formula**: IoU = Intersection / Union for each class, then average

### 2. Dice Coefficient
- **File**: `dice_coefficient.py`
- **Description**: Measures the overlap between predicted and ground truth masks
- **Formula**: Dice = (2 * Intersection) / (Prediction + Ground Truth)

### 3. Pixel Accuracy
- **File**: `pixel_accuracy.py`
- **Description**: Percentage of correctly predicted pixels
- **Formula**: Pixel Accuracy = Correct Pixels / Total Pixels

### 4. Frequency Weighted IoU (FWIoU)
- **File**: `fwiou.py`
- **Description**: Class-balanced IoU weighted by class frequency (pixel proportion) in ground truth
- **Formula**: FWIoU = Σ(IoU_i × w_i), where w_i is the frequency weight of class i
- **Use Case**: Handles class imbalance in road scenes where background classes (buildings, sky) dominate while critical objects (traffic lights, pedestrians) are rare. Gives more weight to frequently appearing classes while still considering all classes.

### 5. Boundary IoU
- **File**: `boundary_iou.py`
- **Description**: IoU calculated only on object boundary/edge regions
- **Formula**: Boundary IoU = IoU computed on boundary pixels only
- **Use Case**: Evaluates segmentation sharpness at object edges. Critical for autonomous driving where edge accuracy (e.g., curbs, obstacles) is more important than interior regions. If edges are blurry or offset, this metric will significantly decrease.

## Usage

### Main Evaluation Script

Run the main evaluation script to calculate all metrics on the test set:

```bash
cd /data/lingyu.li/group_project/evaluation_matrix
python evaluate.py \
    --checkpoint ../UNet_baseline/checkpoints/best.pth \
    --data_root ../data \
    --class_dict ../data/class_dict.csv \
    --batch_size 4 
```

### Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--data_root`: Root directory of dataset (default: `../data`)
- `--class_dict`: Path to class_dict.csv (default: `../data/class_dict.csv`)
- `--batch_size`: Batch size for evaluation (default: 4)
- `--img_size`: Image size (default: 512)
- `--num_workers`: Number of data loading workers (default: 4)
- `--ignore_index`: Class index to ignore, e.g., void class (optional)

### Output

The script will:
1. Print metrics to console
2. Save detailed results to `evaluation_results.txt`

### Using Individual Metrics

You can also import and use individual metrics in your own code:

```python
from miou import calculate_miou, calculate_miou_batch
from dice_coefficient import calculate_mean_dice, calculate_dice_batch
from pixel_accuracy import calculate_pixel_accuracy, calculate_pixel_accuracy_batch
from fwiou import calculate_fwiou, calculate_fwiou_batch
from boundary_iou import calculate_boundary_iou, calculate_boundary_iou_batch

# For single prediction
# Note: num_classes should match your dataset (32 for this project's class_dict.csv)
miou, iou_per_class = calculate_miou(pred_mask, gt_mask, num_classes=32)
mean_dice, dice_per_class = calculate_mean_dice(pred_mask, gt_mask, num_classes=32)
pixel_acc, correct, total = calculate_pixel_accuracy(pred_mask, gt_mask)
fwiou, fwiou_iou_per_class, weights = calculate_fwiou(pred_mask, gt_mask, num_classes=32)
boundary_iou, boundary_iou_per_class = calculate_boundary_iou(pred_mask, gt_mask, num_classes=32, boundary_width=1)

# For batch predictions
miou, iou_per_class = calculate_miou_batch(pred_masks, gt_masks, num_classes=32)
mean_dice, dice_per_class = calculate_dice_batch(pred_masks, gt_masks, num_classes=32)
pixel_acc, correct, total = calculate_pixel_accuracy_batch(pred_masks, gt_masks)
fwiou, fwiou_iou_per_class, weights = calculate_fwiou_batch(pred_masks, gt_masks, num_classes=32)
boundary_iou, boundary_iou_per_class = calculate_boundary_iou_batch(pred_masks, gt_masks, num_classes=32, boundary_width=1)
```
