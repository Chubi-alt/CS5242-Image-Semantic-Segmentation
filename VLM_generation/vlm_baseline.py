"""
VLM Baseline: Generate descriptions for full images
Focus on object counts and relative positions
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm

from vlm_reasoning import VLMReasoner


def create_baseline_prompt(style: str = "detailed") -> str:
    """
    Create prompts for baseline VLM description focusing on object counts and positions.
    
    Args:
        style: Prompt style - "detailed", "count_focused", "position_focused", or "comprehensive"
    
    Returns:
        Prompt string
    """
    prompts = {
        "detailed": (
            "Please provide a detailed description of this image. "
            "Focus on: (1) Count the number of each type of object you can see. "
            "(2) Describe the relative positions of objects (left, right, center, foreground, background). "
            "(3) Describe the spatial relationships between objects (next to, behind, in front of, etc.)."
        ),
        "count_focused": (
            "Count and list all the objects in this image. "
            "For each type of object, specify how many instances you see. "
            "Also describe their approximate locations (left, right, center, top, bottom, foreground, background)."
        ),
        "position_focused": (
            "Describe the spatial layout and relative positions of all objects in this image. "
            "For each object, specify: (1) What it is, (2) Where it is located (left/right/center, "
            "foreground/background, top/middle/bottom), (3) What objects are nearby or adjacent to it."
        ),
        "comprehensive": (
            "Provide a comprehensive scene description including: "
            "(1) A complete inventory: count each type of object (e.g., '3 cars', '2 pedestrians', '1 building'). "
            "(2) Spatial layout: describe where each object is positioned (left, right, center, foreground, background). "
            "(3) Relative positions: explain how objects are positioned relative to each other "
            "(e.g., 'a car is to the left of a building', 'a pedestrian is in front of a car'). "
            "(4) Overall scene structure: describe the general arrangement and composition of the scene."
        ),
        "concise": (
            "Describe this road scene with accurate object counting and spatial relationships. "
            "Format: List each object type with its count (e.g., '2 cars', '1 pedestrian', '3 trees'). "
            "Then briefly describe key spatial relationships (e.g., 'car is left of building', 'pedestrian is on sidewalk'). "
            "Only describe objects you can clearly see. Be precise with counts."
        )
    }
    return prompts.get(style, prompts["detailed"])


def describe_single_image(
    image_path: str,
    vlm_reasoner: VLMReasoner,
    prompt_style: str = "comprehensive",
    custom_prompt: Optional[str] = None,
    max_new_tokens: int = 1024
) -> dict:
    """
    Generate description for a single image.
    
    Args:
        image_path: Path to input image
        vlm_reasoner: VLMReasoner instance
        prompt_style: Style of prompt to use
        custom_prompt: Custom prompt (overrides prompt_style if provided)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with image path and description
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
    vlm_reasoner: VLMReasoner,
    output_dir: str,
    prompt_style: str = "comprehensive",
    custom_prompt: Optional[str] = None,
    max_images: Optional[int] = None,
    max_new_tokens: int = 1024
) -> List[dict]:
    """
    Generate descriptions for all images in a directory.
    
    Args:
        images_dir: Directory containing images
        vlm_reasoner: VLMReasoner instance
        output_dir: Directory to save results
        prompt_style: Style of prompt to use
        custom_prompt: Custom prompt (overrides prompt_style if provided)
        max_images: Maximum number of images to process (None = all)
        max_new_tokens: Maximum tokens to generate per image
    
    Returns:
        List of description dictionaries
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
    
    # Save summary report
    report_path = os.path.join(output_dir, 'description_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VLM Baseline Description Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Successful: {sum(1 for r in results if 'error' not in r)}\n")
        f.write(f"Failed: {sum(1 for r in results if 'error' in r)}\n")
        f.write(f"Prompt style: {prompt_style}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Descriptions\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"\nImage: {result['image_name']}\n")
            f.write("-" * 80 + "\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Description:\n{result['description']}\n")
            f.write("\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - Individual descriptions: {len([r for r in results if 'error' not in r])} files")
    print(f"  - Combined results: {combined_output}")
    print(f"  - Summary report: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='VLM Baseline: Generate descriptions for full images with focus on object counts and positions'
    )
    
    # Input/Output
    parser.add_argument('--mode', type=str, choices=['single', 'directory'], default='directory',
                        help='Processing mode: single image or directory of images')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image (for single mode)')
    parser.add_argument('--images_dir', type=str, default='../data/test',
                        help='Directory containing images (for directory mode)')
    parser.add_argument('--output_dir', type=str, default='./vlm_baseline_results',
                        help='Output directory for results')
    
    # VLM settings
    parser.add_argument('--vlm_model', type=str, default='./Qwen3-VL-4B-Instruct',
                        help='VLM model name or path (default: ./Qwen3-VL-4B-Instruct)')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load VLM model in 4-bit quantization')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load VLM model in 8-bit quantization')
    parser.add_argument('--max_memory_gb', type=float, default=16.0,
                        help='Maximum GPU memory to use in GB (default: 16.0)')
    parser.add_argument('--use_single_gpu', action='store_true', default=True,
                        help='Force using single GPU (default: True)')
    
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
    
    # Initialize VLM
    print("Initializing VLM...")
    # Default to 8-bit if neither specified
    use_4bit = args.load_in_4bit
    use_8bit = args.load_in_8bit
    if not use_4bit and not use_8bit:
        use_8bit = True
        print("No quantization specified. Using 8-bit quantization by default for 16GB GPU constraint.")
    
    vlm_reasoner = VLMReasoner(
        model_name=args.vlm_model,
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
        max_memory_gb=args.max_memory_gb,
        use_single_gpu=args.use_single_gpu
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
