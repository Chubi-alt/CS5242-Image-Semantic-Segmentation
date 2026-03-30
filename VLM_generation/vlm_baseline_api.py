"""
VLM Baseline using API: Generate descriptions for full images using API service
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm

from vlm_api_reasoning import VLMAPIReasoner
from vlm_baseline import create_baseline_prompt


def describe_single_image(
    image_path: str,
    vlm_reasoner: VLMAPIReasoner,
    prompt_style: str = "concise",
    custom_prompt: Optional[str] = None,
    max_new_tokens: int = 1024
) -> dict:
    """
    Generate description for a single image using API.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Get prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = create_baseline_prompt(prompt_style)
    
    # Generate description
    print(f"Generating description for: {os.path.basename(image_path)}")
    description = vlm_reasoner.reason_about_instance(image, prompt, max_new_tokens=max_new_tokens)
    
    return {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'prompt': prompt,
        'description': description,
        'prompt_style': prompt_style
    }


def describe_image_directory(
    images_dir: str,
    vlm_reasoner: VLMAPIReasoner,
    output_dir: str,
    prompt_style: str = "concise",
    custom_prompt: Optional[str] = None,
    max_images: Optional[int] = None,
    max_new_tokens: int = 1024
) -> List[dict]:
    """
    Generate descriptions for all images in a directory using API.
    """
    import glob
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Directory not found: {images_dir}")
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png'))) + \
                  sorted(glob.glob(os.path.join(images_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(images_dir, '*.jpeg')))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    results = []
    for image_path in tqdm(image_files, desc="Generating descriptions"):
        try:
            result = describe_single_image(
                image_path,
                vlm_reasoner,
                prompt_style=prompt_style,
                custom_prompt=custom_prompt,
                max_new_tokens=max_new_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'error': str(e)
            })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    for result in results:
        if 'error' not in result:
            image_name = os.path.splitext(result['image_name'])[0]
            output_file = os.path.join(output_dir, f"{image_name}_description.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save combined results
    combined_output = os.path.join(output_dir, 'all_descriptions.json')
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - Individual descriptions: {len([r for r in results if 'error' not in r])} files")
    print(f"  - Combined results: {combined_output}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='VLM Baseline using API: Generate descriptions for full images'
    )
    
    # Input/Output
    parser.add_argument('--mode', type=str, choices=['single', 'directory'], default='directory',
                        help='Processing mode: single image or directory of images')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image (for single mode)')
    parser.add_argument('--images_dir', type=str, default='../data/vlm_attempt',
                        help='Directory containing images (for directory mode)')
    parser.add_argument('--output_dir', type=str, default='./vlm_baseline_results',
                        help='Output directory for results')
    
    # API settings
    parser.add_argument('--api_provider', type=str, choices=['openai', 'anthropic'], default='openai',
                        help='API provider (default: openai)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name (e.g., gpt-4o, gpt-4-vision-preview, claude-3-opus-20240229)')
    
    # Prompt settings
    parser.add_argument('--prompt_style', type=str,
                        choices=['detailed', 'count_focused', 'position_focused', 'comprehensive', 'concise'],
                        default='concise',
                        help='Style of prompt to use (default: concise)')
    parser.add_argument('--custom_prompt', type=str, default=None,
                        help='Custom prompt (overrides prompt_style if provided)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate (default: 1024)')
    
    # Processing settings
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (for directory mode)')
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process based on mode
    if args.mode == 'single':
        if args.image_path is None:
            parser.error("--image_path is required for single mode")
        
        print(f"\nProcessing single image: {args.image_path}")
        result = describe_single_image(
            args.image_path,
            vlm_reasoner,
            prompt_style=args.prompt_style,
            custom_prompt=args.custom_prompt,
            max_new_tokens=args.max_new_tokens
        )
        
        # Save result
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_file = os.path.join(args.output_dir, f"{image_name}_description.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nDescription saved to {output_file}")
        print("\nDescription:")
        print("-" * 80)
        print(result['description'])
        print("-" * 80)
        
    else:  # directory mode
        if not os.path.exists(args.images_dir):
            parser.error(f"Images directory not found: {args.images_dir}")
        
        print(f"\nProcessing directory: {args.images_dir}")
        results = describe_image_directory(
            args.images_dir,
            vlm_reasoner,
            args.output_dir,
            prompt_style=args.prompt_style,
            custom_prompt=args.custom_prompt,
            max_images=args.max_images,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {args.output_dir}")
    
    print("\nProcessing completed!")


if __name__ == '__main__':
    main()
