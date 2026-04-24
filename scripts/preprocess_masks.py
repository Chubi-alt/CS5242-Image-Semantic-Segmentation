import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def preprocess():
    """
    Pre-converts RGB masks to single-channel index masks to remove 
    CPU bottleneck during training.
    """
    DATA_ROOT = "../data/CamVid"
    # Process both train and val labels
    folders = ['train_labels', 'val_labels', 'test_labels']
    class_dict = pd.read_csv(os.path.join(DATA_ROOT, 'class_dict.csv'))
    
    color_to_index = {
        (r, g, b): idx for idx, (r, g, b) in enumerate(
            zip(class_dict['r'], class_dict['g'], class_dict['b'])
        )
    }

    for folder in folders:
        src_dir = os.path.join(DATA_ROOT, folder)
        # Create a new directory for indexed masks
        dst_dir = os.path.join(DATA_ROOT, folder + '_indexed')
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"[*] Preprocessing {folder}...")
        for filename in tqdm(os.listdir(src_dir)):
            if not filename.endswith('.png'): continue
            
            mask_rgb = cv2.imread(os.path.join(src_dir, filename))
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            
            # Create the 2D index mask
            h, w, _ = mask_rgb.shape
            mask_idx = np.zeros((h, w), dtype=np.uint8)
            
            for color, idx in color_to_index.items():
                match = np.all(mask_rgb == color, axis=-1)
                mask_idx[match] = idx
            
            # Save as a single-channel PNG
            cv2.imwrite(os.path.join(dst_dir, filename), mask_idx)

if __name__ == '__main__':
    preprocess()