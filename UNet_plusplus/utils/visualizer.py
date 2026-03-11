import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_loss_curve(train_losses, val_losses, save_dir="outputs", filename="loss_curve.png"):
    """
    Plots the training and validation loss curves and saves the plot as an image.
    This helps in visually monitoring the model's convergence and checking for overfitting.

    Args:
        train_losses (list of float): A list containing the average training loss per epoch.
        val_losses (list of float): A list containing the average validation loss per epoch.
        save_dir (str): The directory where the image will be saved.
        filename (str): The name of the output image file.
    """
    # Ensure the output directory exists to avoid FileNotFoundError
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # Create a new figure with specific dimensions (width, height)
    plt.figure(figsize=(10, 6))

    # Plot the lines. 
    # 'marker' adds dots to each epoch data point for clearer visibility.
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o', markersize=4)
    plt.plot(val_losses, label='Validation Loss', color='red', marker='s', markersize=4)

    # Add informative title and axis labels
    plt.title('UNet++ Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add a grid to make reading values easier
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend to differentiate between Train and Val
    plt.legend(loc='upper right')

    # Save the figure to the specified path
    # bbox_inches='tight' ensures that labels and titles are not cut off
    plt.savefig(save_path, bbox_inches='tight')
    
    # Close the figure to free up system memory (crucial for long training runs)
    plt.close()
    
    print(f"Loss curve updated and saved to: {save_path}")
def plot_metric_curve(train_values, val_values, metric_name="Loss", save_dir="outputs", filename=None):
    """
    Generic plotting function to visualize any metric (Loss, mIoU, Accuracy) 
    over training epochs.
    
    Args:
        train_values (list): History of values from training phase.
        val_values (list): History of values from validation phase.
        metric_name (str): The name of the metric (e.g., 'mIoU').
        save_dir (str): Directory to save the plot.
        filename (str): If None, defaults to '{metric_name}_curve.png'.
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        # e.g., 'miou_curve.png'
        filename = f"{metric_name.lower().replace(' ', '_')}_curve.png"
    
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    if train_values is not None:
        plt.plot(train_values, label=f'Train {metric_name}', color='blue', marker='o', markersize=4)
    plt.plot(val_values, label=f'Val {metric_name}', color='red', marker='s', markersize=4)

    plt.title(f'UNet++ {metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[*] {metric_name} curve saved to: {save_path}")


# if __name__ == '__main__':
#     """
#     A quick test block to verify that the plotting function works correctly 
#     using dummy data before integrating it into the main training loop.
#     """
#     print("Testing the loss visualization function...")
    
#     # Generate some dummy loss data simulating a learning process
#     dummy_train_loss = [2.5, 1.8, 1.2, 0.9, 0.7, 0.55, 0.45]
#     dummy_val_loss = [2.4, 1.9, 1.4, 1.1, 0.85, 0.75, 0.70]
    
#     # Call the function
#     plot_loss_curve(dummy_train_loss, dummy_val_loss, save_dir="../outputs", filename="test_loss_curve.png")
    
#     print("Test complete! Please check the 'outputs' folder for the generated image.")


def decode_segmap(mask_index, index_to_color):
    """
    Converts a 2D array of class indices into an RGB image for visualization.

    Args:
        mask_index (numpy.ndarray or torch.Tensor): 2D array of shape (H, W) containing class indices.
        index_to_color (dict): Dictionary mapping class index (int) to RGB tuple (R, G, B).

    Returns:
        numpy.ndarray: RGB image of shape (H, W, 3) with type uint8.
    """
    # Ensure mask is a numpy array
    if isinstance(mask_index, torch.Tensor):
        # Detach from computation graph, move to CPU, and convert to numpy
        mask_index = mask_index.detach().cpu().numpy()

    # Initialize empty arrays for R, G, B channels with the same spatial dimensions
    r = np.zeros_like(mask_index, dtype=np.uint8)
    g = np.zeros_like(mask_index, dtype=np.uint8)
    b = np.zeros_like(mask_index, dtype=np.uint8)

    # Iterate through the color mapping and assign RGB values based on class indices
    for idx, color in index_to_color.items():
        # Find coordinates where the mask matches the current class index
        idx_match = (mask_index == idx)
        
        # Assign the corresponding color values
        r[idx_match] = color[0]
        g[idx_match] = color[1]
        b[idx_match] = color[2]

    # Stack the discrete channels along the last axis to form an RGB image
    rgb_mask = np.stack([r, g, b], axis=2)
    return rgb_mask

def visualize_prediction(image, mask_gt, mask_pred, index_to_color, save_dir="outputs", filename="prediction.png"):
    """
    Creates a side-by-side comparison figure of the original image, ground truth, 
    and model prediction, then saves it to the specified directory.

    Args:
        image (torch.Tensor or numpy.ndarray): The input RGB image. 
                                               Expected shape: (3, H, W) if Tensor.
        mask_gt (torch.Tensor or numpy.ndarray): Ground truth mask indices. Shape: (H, W).
        mask_pred (torch.Tensor or numpy.ndarray): Model predicted mask indices. Shape: (H, W).
        index_to_color (dict): Dictionary for decoding indices to RGB colors.
        save_dir (str): Directory to save the output figure.
        filename (str): Name of the output image file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 1. Process the input original image for plotting
    if isinstance(image, torch.Tensor):
        # Convert from PyTorch format (C, H, W) to Matplotlib format (H, W, C)
        img_vis = image.detach().cpu().permute(1, 2, 0).numpy()
        
        # If the image was normalized to [0, 1] in the dataset, scale it back to [0, 255]
        if img_vis.max() <= 1.0:
            img_vis = (img_vis * 255).astype(np.uint8)
    else:
        img_vis = image

    # 2. Decode the index masks back to RGB images using the helper function
    gt_rgb = decode_segmap(mask_gt, index_to_color)
    pred_rgb = decode_segmap(mask_pred, index_to_color)

    # 3. Create the subplot figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Original Image
    axes[0].imshow(img_vis)
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide grid lines and axis ticks

    # Plot Ground Truth Mask
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Plot Model Prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Model Prediction")
    axes[2].axis('off')

    # Adjust layout to prevent title/image overlap and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction visualization saved to: {save_path}")