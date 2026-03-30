"""
Evaluate VLM descriptions using three metrics:
A. Counting Accuracy (MAE/RMSE)
B. Hallucination Check (CHAIR-like metric)
C. LLM-as-a-Judge (GPT-4o evaluation)
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import openai
from tqdm import tqdm


def load_class_dict(class_dict_path: str) -> Dict[str, int]:
    """Load class dictionary and create mapping."""
    df = pd.read_csv(class_dict_path)
    class_names = df['name'].tolist()
    return {name.lower(): idx for idx, name in enumerate(class_names)}


def extract_object_counts(description: str) -> Dict[str, int]:
    """
    Extract object counts from VLM description.
    Handles patterns like "3 cars", "2 pedestrians", "1 building", etc.
    """
    counts = {}
    
    # Pattern to match: number + object name
    # Examples: "3 cars", "2 pedestrians", "1 building", "a car", "an apple"
    patterns = [
        r'(\d+)\s+([a-z]+(?:s|es|ies)?)',  # "3 cars", "2 buildings"
        r'(?:a|an|one)\s+([a-z]+(?:s|es|ies)?)',  # "a car", "an apple"
    ]
    
    description_lower = description.lower()
    
    # First pattern: explicit numbers
    for match in re.finditer(r'(\d+)\s+([a-z]+(?:s|es|ies)?)', description_lower):
        count = int(match.group(1))
        obj_name = match.group(2)
        # Remove plural forms
        obj_name_singular = re.sub(r'(s|es|ies)$', '', obj_name)
        if obj_name_singular not in counts:
            counts[obj_name_singular] = 0
        counts[obj_name_singular] += count
    
    # Second pattern: "a/an/one" = 1
    for match in re.finditer(r'(?:^|\s)(?:a|an|one)\s+([a-z]+(?:s|es|ies)?)', description_lower):
        obj_name = match.group(1)
        obj_name_singular = re.sub(r'(s|es|ies)$', '', obj_name)
        if obj_name_singular not in counts:
            counts[obj_name_singular] = 0
        counts[obj_name_singular] += 1
    
    return counts


def get_mask_classes(mask_path: str, class_dict_path: str) -> set:
    """
    Extract unique class names present in the segmentation mask.
    """
    from PIL import Image
    import numpy as np
    
    # Load mask
    mask = np.array(Image.open(mask_path).convert('RGB'))
    
    # Load class dict RGB values
    df = pd.read_csv(class_dict_path)
    rgb_to_class = {}
    for idx, row in df.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        rgb_to_class[rgb] = row['name'].lower()
    
    # Find unique classes in mask (sample to speed up)
    unique_classes = set()
    h, w = mask.shape[:2]
    # Sample every 10th pixel to speed up
    for i in range(0, h, 10):
        for j in range(0, w, 10):
            rgb = tuple(mask[i, j])
            if rgb in rgb_to_class:
                unique_classes.add(rgb_to_class[rgb])
    
    return unique_classes


def count_instances_in_mask(mask_path: str, class_dict_path: str) -> Dict[str, int]:
    """
    Count instances of each class in the segmentation mask.
    Uses connected components to count distinct instances.
    """
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    
    # Load mask
    mask = np.array(Image.open(mask_path).convert('RGB'))
    
    # Load class dict
    df = pd.read_csv(class_dict_path)
    rgb_to_class = {}
    for idx, row in df.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        rgb_to_class[rgb] = row['name'].lower()
    
    # Create class index mask
    h, w = mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int32)
    class_names = {}
    
    for idx, row in df.iterrows():
        rgb = (int(row['r']), int(row['g']), int(row['b']))
        class_name = row['name'].lower()
        matches = np.all(mask == np.array(rgb), axis=2)
        class_mask[matches] = idx
        class_names[idx] = class_name
    
    # Count instances using connected components
    counts = {}
    for class_idx, class_name in class_names.items():
        if class_name == 'void':
            continue
        binary_mask = (class_mask == class_idx)
        if np.any(binary_mask):
            # Count connected components
            labeled, num_features = ndimage.label(binary_mask)
            counts[class_name] = num_features
    
    return counts


def calculate_counting_accuracy(
    baseline_counts: Dict[str, int],
    mask_counts: Dict[str, int],
    ground_truth_counts: Dict[str, int]
) -> Tuple[float, float, float, float, Dict]:
    """
    Calculate counting accuracy using MAE and RMSE against ground truth.
    
    Args:
        baseline_counts: Object counts from baseline description
        mask_counts: Object counts from mask description
        ground_truth_counts: Actual counts from mask (ground truth)
    
    Returns:
        baseline_mae: MAE for baseline
        mask_mae: MAE for mask description
        baseline_rmse: RMSE for baseline
        mask_rmse: RMSE for mask description
        details: Detailed per-class errors
    """
    # Get all objects that appear in ground truth
    all_objects = set(ground_truth_counts.keys())
    
    baseline_errors = []
    mask_errors = []
    details = {}
    
    for obj in all_objects:
        gt_count = ground_truth_counts.get(obj, 0)
        baseline_count = baseline_counts.get(obj, 0)
        mask_count = mask_counts.get(obj, 0)
        
        baseline_error = abs(baseline_count - gt_count)
        mask_error = abs(mask_count - gt_count)
        
        baseline_errors.append(baseline_error)
        mask_errors.append(mask_error)
        
        details[obj] = {
            'ground_truth': gt_count,
            'baseline': baseline_count,
            'mask': mask_count,
            'baseline_error': baseline_error,
            'mask_error': mask_error
        }
    
    baseline_mae = np.mean(baseline_errors) if baseline_errors else 0.0
    mask_mae = np.mean(mask_errors) if mask_errors else 0.0
    baseline_rmse = np.sqrt(np.mean([e**2 for e in baseline_errors])) if baseline_errors else 0.0
    mask_rmse = np.sqrt(np.mean([e**2 for e in mask_errors])) if mask_errors else 0.0
    
    return baseline_mae, mask_mae, baseline_rmse, mask_rmse, details


def check_hallucination(
    description: str,
    mask_classes: set,
    class_dict: Dict[str, int]
) -> Tuple[int, int, List[str]]:
    """
    Check for hallucinated objects (objects mentioned but not in mask).
    
    Returns:
        total_objects: Total number of objects mentioned
        hallucinated: Number of hallucinated objects
        hallucinated_list: List of hallucinated object names
    """
    counts = extract_object_counts(description)
    total_objects = sum(counts.values())
    
    hallucinated = []
    for obj_name in counts.keys():
        # Check if object matches any class in mask
        is_present = False
        for mask_class in mask_classes:
            # Fuzzy matching
            if (obj_name in mask_class or mask_class in obj_name or
                obj_name.startswith(mask_class[:3]) or mask_class.startswith(obj_name[:3])):
                is_present = True
                break
        
        if not is_present:
            hallucinated.append(obj_name)
    
    return total_objects, len(hallucinated), hallucinated


def llm_as_judge(
    original_image_path: str,
    baseline_description: str,
    mask_description: str,
    api_key: str = None
) -> Dict:
    """
    Use GPT-4o as a judge to evaluate descriptions.
    
    Args:
        original_image_path: Path to original image
        baseline_description: Baseline VLM description
        mask_description: Mask-based VLM description
        api_key: OpenAI API key
    
    Returns:
        Evaluation scores and feedback
    """
    if api_key is None:
        # Try to get from environment
        api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key is None:
        return {
            'baseline_score': None,
            'mask_score': None,
            'feedback': 'OpenAI API key not provided',
            'error': 'API key missing'
        }
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Read image as base64
        import base64
        with open(original_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = f"""请根据原图，对比以下两个描述。

描述 A (Baseline - 仅原始图片):
{baseline_description[:2000]}

描述 B (With Mask - 原始图片+分割掩码):
{mask_description[:2000]}

请重点评估：
1. 物体计数的准确性（哪个描述更准确地数出了物体数量）
2. 空间方位描述（如 'behind', 'left of', 'in front of'）是否符合事实
3. 是否存在无效的重复信息或幻觉（提到不存在的物体）
4. 整体描述的准确性和有用性

请分别为两个描述打分（1-10分），并给出简要评价。格式：
Baseline分数: X/10
Mask分数: Y/10
评价: [你的评价]
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        feedback = response.choices[0].message.content
        
        # Extract scores
        baseline_score = None
        mask_score = None
        
        baseline_match = re.search(r'baseline[^\d]*(\d+)', feedback.lower())
        if baseline_match:
            baseline_score = int(baseline_match.group(1))
        
        mask_match = re.search(r'mask[^\d]*(\d+)', feedback.lower())
        if mask_match:
            mask_score = int(mask_match.group(1))
        
        return {
            'baseline_score': baseline_score,
            'mask_score': mask_score,
            'feedback': feedback
        }
        
    except Exception as e:
        return {
            'baseline_score': None,
            'mask_score': None,
            'feedback': f'Error: {str(e)}',
            'error': str(e)
        }


def load_ground_truth_annotations(gt_file: str) -> Dict:
    """Load ground truth annotations from JSON file."""
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to dictionary keyed by image name
    gt_dict = {}
    for img_data in data.get('images', []):
        img_name = img_data['image_name']
        gt_dict[img_name] = {
            'object_counts': {k.lower(): v for k, v in img_data['object_counts'].items()},
            'present_classes': [c.lower() for c in img_data['present_classes']]
        }
    
    return gt_dict


def evaluate_descriptions(
    baseline_dir: str,
    mask_dir: str,
    mask_results_dir: str,
    class_dict_path: str,
    images_dir: str,
    ground_truth_labels_dir: str = None,
    ground_truth_file: str = None,
    openai_api_key: str = None
) -> Dict:
    """
    Evaluate VLM descriptions using all three metrics.
    
    Args:
        baseline_dir: Directory containing baseline descriptions
        mask_dir: Directory containing mask images
        mask_results_dir: Directory containing mask-based descriptions
        class_dict_path: Path to class_dict.csv
        images_dir: Directory containing original images
        openai_api_key: OpenAI API key for LLM-as-a-Judge
    
    Returns:
        Evaluation results
    """
    class_dict = load_class_dict(class_dict_path)
    
    # Load ground truth annotations if provided
    gt_annotations = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        print(f"Loading ground truth annotations from: {ground_truth_file}")
        gt_annotations = load_ground_truth_annotations(ground_truth_file)
    
    # Get all image files
    baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('_description.json')]
    
    results = []
    all_mae_errors = []
    all_rmse_errors = []
    baseline_hallucinations = []
    mask_hallucinations = []
    llm_scores_baseline = []
    llm_scores_mask = []
    
    for baseline_file in tqdm(baseline_files, desc="Evaluating"):
        # Extract image name
        image_name = baseline_file.replace('_description.json', '.png')
        image_name_base = image_name.replace('.png', '')
        
        # Load descriptions
        baseline_path = os.path.join(baseline_dir, baseline_file)
        mask_desc_path = os.path.join(mask_results_dir, f"{image_name_base}_mask_description.json")
        
        if not os.path.exists(mask_desc_path):
            continue
        
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        with open(mask_desc_path, 'r', encoding='utf-8') as f:
            mask_data = json.load(f)
        
        baseline_desc = baseline_data.get('description', '')
        mask_desc = mask_data.get('description', '')
        
        # Get mask path (predicted mask)
        mask_path = os.path.join(mask_dir, f"{image_name_base}_pred.png")
        if not os.path.exists(mask_path):
            continue
        
        # Determine ground truth source (priority: JSON file > GT labels > predicted mask)
        if gt_annotations and image_name in gt_annotations:
            # Use manual annotations from JSON file
            gt_data = gt_annotations[image_name]
            ground_truth_counts = gt_data['object_counts']
            gt_mask_classes = set(gt_data['present_classes'])
            print(f"  Using manual annotations for {image_name}")
        elif ground_truth_labels_dir:
            # Use ground truth labels as reference
            gt_label_path = os.path.join(ground_truth_labels_dir, f"{image_name_base}_L.png")
            if os.path.exists(gt_label_path):
                gt_mask_path = gt_label_path
                gt_mask_classes = get_mask_classes(gt_label_path, class_dict_path)
                ground_truth_counts = count_instances_in_mask(gt_label_path, class_dict_path)
                print(f"  Using ground truth labels for {image_name}")
            else:
                # Fallback to predicted mask
                gt_mask_path = mask_path
                gt_mask_classes = get_mask_classes(mask_path, class_dict_path)
                ground_truth_counts = count_instances_in_mask(mask_path, class_dict_path)
                print(f"  Warning: GT label not found for {image_name}, using predicted mask")
        else:
            # Use predicted mask as reference
            gt_mask_path = mask_path
            gt_mask_classes = get_mask_classes(mask_path, class_dict_path)
            ground_truth_counts = count_instances_in_mask(mask_path, class_dict_path)
        
        # A. Counting Accuracy
        baseline_counts = extract_object_counts(baseline_desc)
        mask_counts = extract_object_counts(mask_desc)
        
        baseline_mae, mask_mae, baseline_rmse, mask_rmse, count_details = calculate_counting_accuracy(
            baseline_counts, mask_counts, ground_truth_counts
        )
        all_mae_errors.append({'baseline': baseline_mae, 'mask': mask_mae})
        all_rmse_errors.append({'baseline': baseline_rmse, 'mask': mask_rmse})
        
        # B. Hallucination Check (use ground truth classes as reference)
        baseline_total, baseline_hall, baseline_hall_list = check_hallucination(
            baseline_desc, gt_mask_classes, class_dict
        )
        mask_total, mask_hall, mask_hall_list = check_hallucination(
            mask_desc, gt_mask_classes, class_dict
        )
        baseline_hallucinations.append(baseline_hall / max(baseline_total, 1))
        mask_hallucinations.append(mask_hall / max(mask_total, 1))
        
        # C. LLM-as-a-Judge
        original_image_path = os.path.join(images_dir, image_name)
        if os.path.exists(original_image_path):
            llm_result = llm_as_judge(
                original_image_path,
                baseline_desc,
                mask_desc,
                openai_api_key
            )
            if llm_result['baseline_score'] is not None:
                llm_scores_baseline.append(llm_result['baseline_score'])
            if llm_result['mask_score'] is not None:
                llm_scores_mask.append(llm_result['mask_score'])
        else:
            llm_result = {'error': 'Image not found'}
        
        results.append({
            'image_name': image_name,
            'counting_baseline_mae': baseline_mae,
            'counting_mask_mae': mask_mae,
            'counting_baseline_rmse': baseline_rmse,
            'counting_mask_rmse': mask_rmse,
            'counting_details': count_details,
            'baseline_hallucination_rate': baseline_hall / max(baseline_total, 1),
            'mask_hallucination_rate': mask_hall / max(mask_total, 1),
            'baseline_hallucinated_objects': baseline_hall_list,
            'mask_hallucinated_objects': mask_hall_list,
            'llm_evaluation': llm_result
        })
    
    # Aggregate results
    baseline_maes = [e['baseline'] for e in all_mae_errors]
    mask_maes = [e['mask'] for e in all_mae_errors]
    baseline_rmses = [e['baseline'] for e in all_rmse_errors]
    mask_rmses = [e['mask'] for e in all_rmse_errors]
    
    summary = {
        'counting_accuracy': {
            'baseline_mean_mae': np.mean(baseline_maes) if baseline_maes else 0,
            'mask_mean_mae': np.mean(mask_maes) if mask_maes else 0,
            'baseline_mean_rmse': np.mean(baseline_rmses) if baseline_rmses else 0,
            'mask_mean_rmse': np.mean(mask_rmses) if mask_rmses else 0,
            'baseline_std_mae': np.std(baseline_maes) if baseline_maes else 0,
            'mask_std_mae': np.std(mask_maes) if mask_maes else 0,
            'baseline_std_rmse': np.std(baseline_rmses) if baseline_rmses else 0,
            'mask_std_rmse': np.std(mask_rmses) if mask_rmses else 0,
        },
        'hallucination': {
            'baseline_mean_rate': np.mean(baseline_hallucinations) if baseline_hallucinations else 0,
            'mask_mean_rate': np.mean(mask_hallucinations) if mask_hallucinations else 0,
            'baseline_std': np.std(baseline_hallucinations) if baseline_hallucinations else 0,
            'mask_std': np.std(mask_hallucinations) if mask_hallucinations else 0,
        },
        'llm_judge': {
            'baseline_mean_score': np.mean(llm_scores_baseline) if llm_scores_baseline else None,
            'mask_mean_score': np.mean(llm_scores_mask) if llm_scores_mask else None,
            'baseline_std': np.std(llm_scores_baseline) if llm_scores_baseline else None,
            'mask_std': np.std(llm_scores_mask) if llm_scores_mask else None,
        },
        'detailed_results': results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VLM descriptions using counting accuracy, hallucination check, and LLM-as-a-Judge'
    )
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Directory containing baseline descriptions')
    parser.add_argument('--mask_results_dir', type=str, required=True,
                        help='Directory containing mask-based descriptions')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing segmentation masks')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing original images')
    parser.add_argument('--ground_truth_file', type=str, default=None,
                        help='Path to JSON file with manual ground truth annotations (highest priority)')
    parser.add_argument('--ground_truth_labels_dir', type=str, default=None,
                        help='Directory containing ground truth label images (if provided and no --ground_truth_file, will use as reference)')
    parser.add_argument('--class_dict', type=str, default='../data/class_dict.csv',
                        help='Path to class_dict.csv')
    parser.add_argument('--output', type=str, default='./vlm_evaluation_results.json',
                        help='Output file for evaluation results')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key for LLM-as-a-Judge (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    
    print("Starting evaluation...")
    if args.ground_truth_file:
        print(f"Using manual ground truth annotations from: {args.ground_truth_file}")
    elif args.ground_truth_labels_dir:
        print(f"Using ground truth labels from: {args.ground_truth_labels_dir}")
    else:
        print("Using predicted masks as reference (consider using --ground_truth_file or --ground_truth_labels_dir for more accurate evaluation)")
    
    results = evaluate_descriptions(
        baseline_dir=args.baseline_dir,
        mask_dir=args.mask_dir,
        mask_results_dir=args.mask_results_dir,
        class_dict_path=args.class_dict,
        images_dir=args.images_dir,
        ground_truth_labels_dir=args.ground_truth_labels_dir,
        ground_truth_file=args.ground_truth_file,
        openai_api_key=api_key
    )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    
    print("\nA. Counting Accuracy (compared to ground truth from mask):")
    print(f"  Baseline MAE: {results['counting_accuracy']['baseline_mean_mae']:.3f} ± {results['counting_accuracy']['baseline_std_mae']:.3f}")
    print(f"  Mask MAE:     {results['counting_accuracy']['mask_mean_mae']:.3f} ± {results['counting_accuracy']['mask_std_mae']:.3f}")
    print(f"  Baseline RMSE: {results['counting_accuracy']['baseline_mean_rmse']:.3f} ± {results['counting_accuracy']['baseline_std_rmse']:.3f}")
    print(f"  Mask RMSE:     {results['counting_accuracy']['mask_mean_rmse']:.3f} ± {results['counting_accuracy']['mask_std_rmse']:.3f}")
    
    print("\nB. Hallucination Rate:")
    print(f"  Baseline: {results['hallucination']['baseline_mean_rate']:.3f} ± {results['hallucination']['baseline_std']:.3f}")
    print(f"  With Mask: {results['hallucination']['mask_mean_rate']:.3f} ± {results['hallucination']['mask_std']:.3f}")
    
    if results['llm_judge']['baseline_mean_score'] is not None:
        print("\nC. LLM-as-a-Judge Scores:")
        print(f"  Baseline: {results['llm_judge']['baseline_mean_score']:.2f} ± {results['llm_judge']['baseline_std']:.2f}")
        print(f"  With Mask: {results['llm_judge']['mask_mean_score']:.2f} ± {results['llm_judge']['mask_std']:.2f}")
    else:
        print("\nC. LLM-as-a-Judge: Not available (API key not provided)")
    
    print(f"\nDetailed results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
