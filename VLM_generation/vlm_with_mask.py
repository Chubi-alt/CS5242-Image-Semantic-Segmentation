"""
VLM with Segmentation Mask: Generate descriptions using original image + UNet mask
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
import torch
import pandas as pd

# Add UNet_baseline to path
unet_baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UNet_baseline'))
if unet_baseline_path not in sys.path:
    sys.path.insert(0, unet_baseline_path)

from unet_model import UNet
from dataset import SegmentationDataset

from vlm_reasoning import VLMReasoner


def get_checkpoint_num_classes(checkpoint):
    """Infer the model output classes from a saved checkpoint."""
    state_dict = checkpoint['model_state_dict']
    return state_dict['outc.bias'].shape[0]


def class_to_rgb_mask(class_mask, class_dict):
    """Convert class index mask to RGB mask"""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, row in class_dict.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        matches = (class_mask == idx)
        rgb_mask[matches] = rgb
    
    return rgb_mask


def generate_mask_with_unet(
    image_path: str,
    unet_model: torch.nn.Module,
    device: torch.device,
    class_dict: pd.DataFrame,
    transform=None
):
    """
    Generate segmentation mask using UNet model.
    
    Args:
        image_path: Path to input image
        unet_model: Trained UNet model
        device: Device to run on
        class_dict: Class dictionary for RGB conversion
        transform: Optional transform for preprocessing
    
    Returns:
        RGB mask as numpy array [H, W, 3]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Apply transform if provided
    if transform:
        transformed = transform(image=image_np)
        image_tensor = transformed['image']
    else:
        # Simple preprocessing: normalize and convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    # Generate mask
    unet_model.eval()
    with torch.no_grad():
        outputs = unet_model(image_tensor)
        preds = torch.argmax(outputs, dim=1)  # [1, H, W]
        pred_mask = preds[0].cpu().numpy()  # [H, W]
    
    # Convert to RGB mask
    rgb_mask = class_to_rgb_mask(pred_mask, class_dict)
    
    return rgb_mask


def create_mask_prompt(style: str = "comprehensive") -> str:
    """
    Create prompts for VLM with segmentation mask.
    
    Args:
        style: Prompt style
    
    Returns:
        Prompt string
    """
    prompts = {
        "detailed": (
            "I have provided you with an original image and its semantic segmentation mask. "
            "The mask shows different regions colored by their semantic classes. "
            "Please provide a detailed description of this scene focusing on: "
            "(1) Count the number of each type of object you can see in the mask. "
            "(2) Describe the relative positions of objects (left, right, center, foreground, background). "
            "(3) Describe the spatial relationships between objects based on the segmentation mask."
        ),
        "comprehensive": (
            "I have provided you with an original image and its semantic segmentation mask. "
            "The mask shows different regions colored by their semantic classes. "
            "Please provide a comprehensive scene description including: "
            "(1) A complete inventory: count each type of object visible in the mask "
            "(e.g., '3 cars', '2 pedestrians', '1 building'). "
            "(2) Spatial layout: describe where each object is positioned according to the mask "
            "(left, right, center, foreground, background). "
            "(3) Relative positions: explain how objects are positioned relative to each other "
            "based on the segmentation mask (e.g., 'a car is to the left of a building', "
            "'a pedestrian is in front of a car'). "
            "(4) Overall scene structure: describe the general arrangement and composition "
            "of the scene as shown in the segmentation mask."
        ),
        "mask_focused": (
            "I have provided you with an original image and its semantic segmentation mask. "
            "The mask shows different colored regions representing different semantic classes. "
            "Please analyze the segmentation mask and describe: "
            "(1) What objects and regions are present in the mask? "
            "(2) How are these objects distributed spatially? "
            "(3) What are the relationships between different segmented regions? "
            "(4) How does the segmentation mask help understand the scene structure?"
        ),
        "concise": (
            "You have been given an original image and its semantic segmentation mask. "
            "The mask uses different colors to represent different object classes. "
            "Describe the scene based on what you see in the mask. "
            "Format: List each object type with its count (e.g., '2 cars', '1 pedestrian', '3 trees'). "
            "Then briefly describe key spatial relationships (e.g., 'car is left of building'). "
            "IMPORTANT: Only describe objects that appear in the segmentation mask. "
            "Count instances accurately by identifying distinct colored regions in the mask."
        )
    }
    return prompts.get(style, prompts["comprehensive"])


def reason_with_mask(
    image_path: str,
    mask_image: np.ndarray,
    vlm_reasoner: VLMReasoner,
    prompt: str,
    max_new_tokens: int = 1024
) -> str:
    """
    Use VLM to reason about image with segmentation mask.
    For Qwen3-VL, we can pass multiple images.
    
    Args:
        image_path: Path to original image
        mask_image: Segmentation mask as numpy array [H, W, 3]
        vlm_reasoner: VLMReasoner instance
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated description
    """
    if vlm_reasoner.model is None or vlm_reasoner.processor is None:
        return "VLM model not available"
    
    try:
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Convert mask to PIL Image
        mask_pil = Image.fromarray(mask_image)
        
        # Check if processor supports multiple images
        processor_type = str(type(vlm_reasoner.processor)).lower()
        is_qwen = 'qwen' in processor_type
        
        if is_qwen:
            # Qwen3-VL can handle multiple images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_image},
                        {"type": "image", "image": mask_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            if hasattr(vlm_reasoner.processor, 'apply_chat_template'):
                text = vlm_reasoner.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if hasattr(vlm_reasoner.processor, 'process_queries'):
                    image_inputs, video_inputs = vlm_reasoner.processor.process_queries(
                        messages, vlm_reasoner.model.config
                    )
                    inputs = vlm_reasoner.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = vlm_reasoner.processor(
                        text=[text],
                        images=[original_image, mask_pil],
                        padding=True,
                        return_tensors="pt",
                    )
            else:
                # Direct processing
                inputs = vlm_reasoner.processor(
                    text=[prompt],
                    images=[original_image, mask_pil],
                    padding=True,
                    return_tensors="pt",
                )
        else:
            # For other models, concatenate images or use single image
            # Create a side-by-side composite image
            width, height = original_image.size
            composite = Image.new('RGB', (width * 2, height))
            composite.paste(original_image, (0, 0))
            composite.paste(mask_pil, (width, 0))
            
            inputs = vlm_reasoner.processor(
                text=[prompt],
                images=[composite],
                padding=True,
                return_tensors="pt",
            )
        
        # Move inputs to device - ensure they match model's device
        model_device = next(vlm_reasoner.model.parameters()).device
        inputs_on_device = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs_on_device[key] = value.to(model_device)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                processed_value = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        processed_value.append(item.to(model_device))
                    else:
                        processed_value.append(item)
                inputs_on_device[key] = type(value)(processed_value) if isinstance(value, tuple) else processed_value
            else:
                inputs_on_device[key] = value
        
        inputs = inputs_on_device
        
        # Generate response
        with torch.no_grad():
            output = vlm_reasoner.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            # Decode response
            if is_qwen:
                if hasattr(vlm_reasoner.processor, 'batch_decode'):
                    generated_text = vlm_reasoner.processor.batch_decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    # Remove input prompt if present
                    if prompt in generated_text:
                        generated_text = generated_text.split(prompt, 1)[-1].strip()
                else:
                    generated_text = vlm_reasoner.processor.decode(
                        output[0],
                        skip_special_tokens=True
                    )
            else:
                # For other models
                if hasattr(vlm_reasoner.processor, 'batch_decode'):
                    generated_text = vlm_reasoner.processor.batch_decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    generated_text = vlm_reasoner.processor.decode(
                        output[0],
                        skip_special_tokens=True
                    )
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error in VLM reasoning: {e}")
        return f"Error: {str(e)}"


def process_vlm_attempt_with_mask(
    images_dir: str,
    vlm_reasoner: VLMReasoner,
    output_dir: str,
    class_dict_path: str,
    prompt_style: str = "comprehensive",
    unet_checkpoint: str = None,
    mask_dir: str = None,
    device: torch.device = None
):
    """
    Process vlm_attempt dataset: use existing masks or generate with UNet, then VLM descriptions.
    
    Args:
        images_dir: Directory containing images
        vlm_reasoner: VLMReasoner instance
        output_dir: Output directory for results
        class_dict_path: Path to class_dict.csv
        prompt_style: Prompt style
        unet_checkpoint: Path to UNet checkpoint (required if mask_dir not provided)
        mask_dir: Directory containing pre-generated masks (optional, if provided, will use these instead of generating)
        device: Device for UNet inference
    """
    import glob
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine whether to use existing masks or generate new ones
    use_existing_masks = mask_dir is not None and os.path.exists(mask_dir)
    
    if use_existing_masks:
        print(f"Using existing masks from: {mask_dir}")
    else:
        if unet_checkpoint is None:
            raise ValueError("Either mask_dir or unet_checkpoint must be provided")
        # Load UNet model
        print(f"Loading UNet checkpoint from {unet_checkpoint}...")
        checkpoint = torch.load(unet_checkpoint, map_location=device)
        num_classes = get_checkpoint_num_classes(checkpoint)
        class_dict = pd.read_csv(class_dict_path)
        
        unet_model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        unet_model = unet_model.to(device)
        unet_model.eval()
        print(f"UNet model loaded. Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Get image files
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    print(f"Processing {len(image_files)} images...")
    
    # Get prompt
    prompt = create_mask_prompt(prompt_style)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for image_path in tqdm(image_files, desc="Processing"):
        try:
            image_name = os.path.basename(image_path)
            print(f"\nProcessing: {image_name}")
            
            # Get mask
            if use_existing_masks:
                # Load existing mask
                mask_name = image_name.replace('.png', '_pred.png')
                mask_path = os.path.join(mask_dir, mask_name)
                if not os.path.exists(mask_path):
                    print(f"  Warning: Mask not found: {mask_path}, skipping...")
                    results.append({
                        'image_path': image_path,
                        'image_name': image_name,
                        'error': f'Mask not found: {mask_path}'
                    })
                    continue
                print("  Loading existing segmentation mask...")
                mask_image = Image.open(mask_path).convert('RGB')
                mask = np.array(mask_image)
            else:
                # Generate mask with UNet
                print("  Generating segmentation mask...")
                mask = generate_mask_with_unet(
                    image_path, unet_model, device, class_dict
                )
            
            # Generate description with VLM
            print("  Generating VLM description...")
            description = reason_with_mask(
                image_path, mask, vlm_reasoner, prompt, max_new_tokens=1024
            )
            
            result = {
                'image_path': image_path,
                'image_name': image_name,
                'prompt': prompt,
                'description': description,
                'prompt_style': prompt_style
            }
            results.append(result)
            
            # Save individual result
            image_name_base = os.path.splitext(image_name)[0]
            output_file = os.path.join(output_dir, f"{image_name_base}_mask_description.json")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Warning: Failed to save {output_file}: {e}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'error': str(e)
            })
    
    # Save combined results
    combined_output = os.path.join(output_dir, 'all_mask_descriptions.json')
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - Individual descriptions: {len([r for r in results if 'error' not in r])} files")
    print(f"  - Combined results: {combined_output}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate VLM descriptions using UNet segmentation masks'
    )
    
    # Input/Output
    parser.add_argument('--images_dir', type=str, default='../data/vlm_attempt',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./vlm_mask_results',
                        help='Output directory for results')
    parser.add_argument('--unet_checkpoint', type=str, default=None,
                        help='Path to UNet checkpoint (required if --mask_dir not provided)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing pre-generated masks (if provided, will use these instead of generating)')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    
    # VLM settings
    parser.add_argument('--vlm_model', type=str, default='./Qwen3-VL-4B-Instruct',
                        help='VLM model name or path')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load VLM model in 4-bit quantization')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load VLM model in 8-bit quantization')
    parser.add_argument('--max_memory_gb', type=float, default=16.0,
                        help='Maximum GPU memory to use in GB')
    parser.add_argument('--use_single_gpu', action='store_true', default=True,
                        help='Force using single GPU')
    
    # Prompt settings
    parser.add_argument('--prompt_style', type=str,
                        choices=['detailed', 'comprehensive', 'mask_focused', 'concise'],
                        default='concise',
                        help='Style of prompt to use (default: concise)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize VLM
    print("Initializing VLM...")
    use_4bit = args.load_in_4bit
    use_8bit = args.load_in_8bit
    if not use_4bit and not use_8bit:
        use_8bit = True
        print("Using 8-bit quantization by default for 16GB GPU constraint.")
    
    vlm_reasoner = VLMReasoner(
        model_name=args.vlm_model,
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
        max_memory_gb=args.max_memory_gb,
        use_single_gpu=args.use_single_gpu
    )
    
    # Process images
    results = process_vlm_attempt_with_mask(
        images_dir=args.images_dir,
        vlm_reasoner=vlm_reasoner,
        output_dir=args.output_dir,
        class_dict_path=args.class_dict,
        prompt_style=args.prompt_style,
        unet_checkpoint=args.unet_checkpoint,
        mask_dir=args.mask_dir,
        device=device
    )
    
    print(f"\nProcessing completed! Processed {len(results)} images.")


if __name__ == '__main__':
    main()
