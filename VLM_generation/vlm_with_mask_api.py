"""
VLM with Mask using API: Generate descriptions using original image + UNet mask via API
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
import glob

from vlm_api_reasoning import VLMAPIReasoner
from vlm_with_mask import create_mask_prompt


def process_vlm_attempt_with_mask_api(
    images_dir: str,
    vlm_reasoner: VLMAPIReasoner,
    output_dir: str,
    mask_dir: str,
    prompt_style: str = "concise",
    max_new_tokens: int = 1024
):
    """
    Process vlm_attempt dataset using API: use existing masks, then VLM descriptions.
    """
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
            image_name_base = image_name.replace('.png', '')
            print(f"\nProcessing: {image_name}")
            
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
            
            print("  Loading segmentation mask...")
            mask_image = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask_image)
            
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            
            # Generate description with VLM API (multiple images)
            print("  Generating VLM description via API...")
            description = vlm_reasoner.reason_with_multiple_images(
                [original_image, mask_image],
                prompt,
                max_new_tokens=max_new_tokens
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
        description='Generate VLM descriptions using UNet segmentation masks via API'
    )
    
    # Input/Output
    parser.add_argument('--images_dir', type=str, default='../data/vlm_attempt',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./vlm_mask_results',
                        help='Output directory for results')
    parser.add_argument('--mask_dir', type=str, default='../UNet_baseline/test_results',
                        help='Directory containing pre-generated masks')
    
    # API settings
    parser.add_argument('--api_provider', type=str, choices=['openai', 'anthropic'], default='openai',
                        help='API provider (default: openai)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name (e.g., gpt-4o, gpt-4-vision-preview, claude-3-opus-20240229)')
    
    # Prompt settings
    parser.add_argument('--prompt_style', type=str,
                        choices=['detailed', 'comprehensive', 'mask_focused', 'concise'],
                        default='concise',
                        help='Style of prompt to use (default: concise)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    # Initialize API reasoner
    print(f"Initializing {args.api_provider} API reasoner...")
    print(f"Model: {args.model_name}")
    vlm_reasoner = VLMAPIReasoner(
        api_provider=args.api_provider,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_new_tokens
    )
    
    # Process images
    results = process_vlm_attempt_with_mask_api(
        images_dir=args.images_dir,
        vlm_reasoner=vlm_reasoner,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_new_tokens
    )
    
    print(f"\nProcessing completed! Processed {len(results)} images.")


if __name__ == '__main__':
    main()
