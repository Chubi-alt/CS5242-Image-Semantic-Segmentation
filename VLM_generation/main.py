"""
Main script for Stage 2: VLM Multimodal Alignment & Generation
Integrates mask-guided instance isolation, VLM reasoning, and evaluation.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'UNet_baseline'))

from mask_guided_isolation import (
    extract_all_instances,
    load_mask_from_rgb,
    save_isolated_instances
)
from vlm_reasoning import (
    VLMReasoner,
    reason_about_all_instances,
    create_targeted_prompts
)
from evaluation import VLMEvaluator


def resolve_existing_path(path: str, description: str) -> str:
    """Resolve and validate an input path."""
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def process_single_image(
    image_path: str,
    mask_path: str,
    class_dict_path: str,
    vlm_reasoner: VLMReasoner,
    output_dir: str,
    padding: int = 10,
    min_area: int = 100,
    target_classes: list = None,
    exclude_classes: list = None,
    task: str = "description"
):
    """
    Process a single image: extract instances, reason about them, and evaluate.
    
    Args:
        image_path: Path to input image
        mask_path: Path to segmentation mask (RGB-encoded)
        class_dict_path: Path to class_dict.csv
        vlm_reasoner: VLMReasoner instance
        output_dir: Output directory for results
        padding: Padding around instances
        min_area: Minimum area for valid instances
        target_classes: List of class indices to process (None = all)
        exclude_classes: List of class indices to exclude
        task: Reasoning task type
    """
    image_path = resolve_existing_path(image_path, 'Input image')
    mask_path = resolve_existing_path(mask_path, 'Segmentation mask')
    class_dict_path = resolve_existing_path(class_dict_path, 'Class dictionary')

    # Load image and mask
    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    
    print(f"Loading mask: {mask_path}")
    class_mask, class_dict = load_mask_from_rgb(mask_path, class_dict_path)
    
    # Extract instances
    print("Extracting instances...")
    instances = extract_all_instances(
        image,
        class_mask,
        class_indices=target_classes,
        padding=padding,
        min_area=min_area,
        exclude_classes=exclude_classes or []
    )
    
    print(f"Found {sum(len(v) for v in instances.values())} instances across {len(instances)} classes")
    
    # Save isolated instances
    instance_dir = os.path.join(output_dir, 'isolated_instances')
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_isolated_instances(instances, instance_dir, base_filename, class_dict)
    
    # Perform VLM reasoning on isolated instances
    print("Performing VLM reasoning on isolated instances...")
    isolated_responses = reason_about_all_instances(
        instances, vlm_reasoner, class_dict, task=task
    )
    
    # Perform VLM reasoning on raw image (baseline)
    print("Performing VLM reasoning on raw image (baseline)...")
    baseline_responses = {}
    for class_idx, instance_list in instances.items():
        class_name = class_dict.get(class_idx, f"class_{class_idx}")
        prompt = create_targeted_prompts(class_name, task)
        
        # Use full image for baseline
        baseline_response = vlm_reasoner.reason_about_instance(image, prompt)
        
        for inst_idx, (_, bbox) in enumerate(instance_list):
            baseline_responses[(class_idx, inst_idx)] = {
                'response': baseline_response,
                'bbox': bbox,
                'class_name': class_name
            }
    
    # Evaluate
    print("Evaluating results...")
    evaluator = VLMEvaluator()
    eval_dir = os.path.join(output_dir, 'evaluation')
    evaluator.generate_comparison_report(
        eval_dir, isolated_responses, baseline_responses
    )
    
    # Save responses
    responses_dir = os.path.join(output_dir, 'responses')
    os.makedirs(responses_dir, exist_ok=True)
    
    import json
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            # Convert tuple to list for JSON serialization
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            # Convert dict: tuple keys become strings, other keys and values are converted
            result = {}
            for key, value in obj.items():
                # Convert key to string if it's a tuple or other non-string type
                if isinstance(key, tuple):
                    # Convert tuple to string representation for JSON
                    converted_key = str([convert_to_serializable(item) for item in key])
                elif isinstance(key, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    converted_key = int(key)
                elif not isinstance(key, (str, int, float, bool)) and key is not None:
                    converted_key = str(key)
                else:
                    converted_key = key
                
                converted_value = convert_to_serializable(value)
                result[converted_key] = converted_value
            return result
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    # Convert responses to serializable format
    isolated_serializable = convert_to_serializable(isolated_responses)
    baseline_serializable = convert_to_serializable(baseline_responses)
    
    with open(os.path.join(responses_dir, f'{base_filename}_isolated.json'), 'w') as f:
        json.dump(isolated_serializable, f, indent=2, ensure_ascii=False)
    with open(os.path.join(responses_dir, f'{base_filename}_baseline.json'), 'w') as f:
        json.dump(baseline_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}")
    
    return isolated_responses, baseline_responses


def process_test_set(
    test_images_dir: str,
    test_masks_dir: str,
    class_dict_path: str,
    vlm_reasoner: VLMReasoner,
    output_dir: str,
    padding: int = 10,
    min_area: int = 100,
    target_classes: list = None,
    exclude_classes: list = None,
    task: str = "description",
    max_images: int = None,
    mask_suffix: str = '_pred.png'
):
    """
    Process entire test set.
    
    Args:
        test_images_dir: Directory containing test images
        test_masks_dir: Directory containing test masks
        class_dict_path: Path to class_dict.csv
        vlm_reasoner: VLMReasoner instance
        output_dir: Output directory
        padding: Padding around instances
        min_area: Minimum area for valid instances
        target_classes: List of class indices to process
        exclude_classes: List of class indices to exclude
        task: Reasoning task type
        max_images: Maximum number of images to process (None = all)
    """
    import glob

    test_images_dir = resolve_existing_path(test_images_dir, 'Test image directory')
    test_masks_dir = resolve_existing_path(test_masks_dir, 'Test mask directory')
    class_dict_path = resolve_existing_path(class_dict_path, 'Class dictionary')
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(test_images_dir, '*.png')))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    all_isolated_responses = {}
    all_baseline_responses = {}
    
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing {os.path.basename(image_path)}")
        
        # Find corresponding mask
        image_name = os.path.basename(image_path)
        mask_name = image_name.replace('.png', mask_suffix)
        mask_path = os.path.join(test_masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {image_name}, skipping...")
            continue
        
        # Process image
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
        isolated, baseline = process_single_image(
            image_path, mask_path, class_dict_path, vlm_reasoner,
            image_output_dir, padding, min_area, target_classes,
            exclude_classes, task
        )
        
        # Aggregate responses
        for key, value in isolated.items():
            all_isolated_responses[(os.path.splitext(image_name)[0], *key)] = value
        for key, value in baseline.items():
            all_baseline_responses[(os.path.splitext(image_name)[0], *key)] = value
    
    # Generate overall evaluation report
    print("\nGenerating overall evaluation report...")
    evaluator = VLMEvaluator()
    overall_eval_dir = os.path.join(output_dir, 'overall_evaluation')
    evaluator.generate_comparison_report(
        overall_eval_dir, all_isolated_responses, all_baseline_responses
    )
    
    print(f"\nAll results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: VLM Multimodal Alignment & Generation'
    )
    
    # Input/Output
    parser.add_argument('--mode', type=str, choices=['single', 'test_set'], default='single',
                        help='Processing mode: single image or entire test set')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to input image (for single mode)')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Path to segmentation mask (for single mode)')
    parser.add_argument('--test_images_dir', type=str, default='../data/test',
                        help='Directory containing test images (for test_set mode)')
    parser.add_argument('--test_masks_dir', type=str, default='../data/test_labels',
                        help='Directory containing test masks (for test_set mode)')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    parser.add_argument('--output_dir', type=str, default='./vlm_results',
                        help='Output directory for results')
    parser.add_argument('--mask_suffix', type=str, default='_pred.png',
                        help='Mask filename suffix for test_set mode (default: _pred.png)')
    
    # VLM settings
    parser.add_argument('--vlm_model', type=str, default='./Qwen3-VL-4B-Instruct',
                        help='VLM model name or path (default: ./Qwen3-VL-4B-Instruct)')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load VLM model in 4-bit quantization (most memory efficient)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load VLM model in 8-bit quantization (balanced)')
    parser.add_argument('--max_memory_gb', type=float, default=16.0,
                        help='Maximum GPU memory to use in GB (default: 16.0)')
    parser.add_argument('--use_single_gpu', action='store_true', default=True,
                        help='Force using single GPU (default: True)')
    
    # Instance extraction settings
    parser.add_argument('--padding', type=int, default=10,
                        help='Padding around instances (pixels)')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Minimum area for valid instances (pixels)')
    parser.add_argument('--target_classes', type=int, nargs='+', default=None,
                        help='List of class indices to process (None = all)')
    parser.add_argument('--exclude_classes', type=int, nargs='+', default=None,
                        help='List of class indices to exclude (e.g., void/background)')
    
    # Reasoning settings
    parser.add_argument('--task', type=str, default='description',
                        choices=['description', 'attributes', 'context', 'detailed'],
                        help='Type of reasoning task')
    
    # Test set settings
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (for test_set mode)')
    
    args = parser.parse_args()
    
    # Initialize VLM
    print("Initializing VLM...")
    # If neither 4bit nor 8bit is specified, use 8bit by default for 16GB constraint
    use_4bit = args.load_in_4bit
    use_8bit = args.load_in_8bit
    if not use_4bit and not use_8bit:
        # Default to 8-bit quantization for 16GB GPU constraint
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
        if args.image_path is None or args.mask_path is None:
            parser.error("--image_path and --mask_path are required for single mode")
        
        process_single_image(
            args.image_path,
            args.mask_path,
            args.class_dict,
            vlm_reasoner,
            args.output_dir,
            args.padding,
            args.min_area,
            args.target_classes,
            args.exclude_classes,
            args.task
        )
    else:  # test_set mode
        process_test_set(
            args.test_images_dir,
            args.test_masks_dir,
            args.class_dict,
            vlm_reasoner,
            args.output_dir,
            args.padding,
            args.min_area,
            args.target_classes,
            args.exclude_classes,
            args.task,
            args.max_images,
            args.mask_suffix
        )
    
    print("\nProcessing completed!")


if __name__ == '__main__':
    main()
