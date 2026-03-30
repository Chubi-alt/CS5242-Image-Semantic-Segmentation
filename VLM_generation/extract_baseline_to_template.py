"""
Extract object counts and classes from baseline VLM descriptions
and populate the ground truth template for manual correction.
"""

import json
import os
import re
from collections import defaultdict


def extract_object_counts(description: str) -> dict:
    """
    Extract object counts from VLM description.
    Handles patterns like "3 cars", "2 pedestrians", "1 building", etc.
    """
    counts = {}
    
    description_lower = description.lower()
    
    # Pattern 1: explicit numbers like "3 cars", "2 pedestrians"
    for match in re.finditer(r'(\d+)\s+([a-z]+(?:s|es|ies)?)', description_lower):
        count = int(match.group(1))
        obj_name = match.group(2)
        # Remove plural forms
        obj_name_singular = re.sub(r'(s|es|ies)$', '', obj_name)
        if obj_name_singular not in counts:
            counts[obj_name_singular] = 0
        counts[obj_name_singular] += count
    
    # Pattern 2: "a/an/one" = 1
    for match in re.finditer(r'(?:^|\s)(?:a|an|one)\s+([a-z]+(?:s|es|ies)?)', description_lower):
        obj_name = match.group(1)
        obj_name_singular = re.sub(r'(s|es|ies)$', '', obj_name)
        if obj_name_singular not in counts:
            counts[obj_name_singular] = 0
        counts[obj_name_singular] += 1
    
    return counts


def normalize_class_name(name: str) -> str:
    """Normalize class name to match template format."""
    name = name.lower().strip()
    # Common mappings
    mappings = {
        'bus': 'truck_bus',
        'truck': 'truck_bus',
        'person': 'pedestrian',
        'people': 'pedestrian',
        'bicycle': 'bicyclist',
        'bike': 'bicyclist',
        'motorcycle': 'motorcyclescooter',
        'scooter': 'motorcyclescooter',
        'suv': 'suvpickuptruck',
        'pickup': 'suvpickuptruck',
        'pickup truck': 'suvpickuptruck',
        'sign': 'signsymbol',
        'pole': 'column_pole',
        'column': 'column_pole',
        'traffic light': 'trafficlight',
        'traffic cone': 'trafficcone',
        'lane marking': 'lanemkgsdriv',
        'road marking': 'lanemkgsdriv',
        'vegetation': 'vegetationmisc',
        'misc text': 'misc_text',
        'text': 'misc_text',
    }
    
    if name in mappings:
        return mappings[name]
    
    # Remove spaces and special characters
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name


def get_all_classes_from_description(description: str) -> list:
    """Extract all mentioned object classes from description."""
    description_lower = description.lower()
    
    # Common class names to look for
    class_keywords = [
        'car', 'truck', 'bus', 'pedestrian', 'person', 'people', 'bicyclist', 'bicycle', 'bike',
        'motorcycle', 'scooter', 'suv', 'pickup', 'building', 'tree', 'trees',
        'traffic light', 'trafficlight', 'traffic cone', 'trafficcone',
        'sign', 'fence', 'pole', 'column', 'road', 'sidewalk', 'sky',
        'vegetation', 'wall', 'bridge', 'tunnel', 'archway', 'train',
        'animal', 'child', 'cart', 'luggage', 'pram', 'moving'
    ]
    
    found_classes = set()
    for keyword in class_keywords:
        # Look for keyword in description
        pattern = r'\b' + re.escape(keyword) + r'\w*\b'
        if re.search(pattern, description_lower):
            normalized = normalize_class_name(keyword)
            found_classes.add(normalized)
    
    return sorted(list(found_classes))


def populate_template_from_baseline(baseline_dir: str, template_file: str, output_file: str):
    """Extract counts from baseline descriptions and populate template."""
    
    # Load template
    with open(template_file, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    # Process each image in template
    for img_data in template['images']:
        image_name = img_data['image_name']
        image_name_base = image_name.replace('.png', '')
        
        # Load baseline description
        baseline_file = os.path.join(baseline_dir, f"{image_name_base}_description.json")
        
        if not os.path.exists(baseline_file):
            print(f"Warning: Baseline file not found for {image_name}")
            continue
        
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        description = baseline_data.get('description', '')
        
        # Extract counts
        extracted_counts = extract_object_counts(description)
        
        # Normalize and map to template format
        template_counts = {}
        for template_key in img_data['object_counts'].keys():
            # Try to find matching extracted count
            count = 0
            for extracted_key, extracted_value in extracted_counts.items():
                normalized_extracted = normalize_class_name(extracted_key)
                if normalized_extracted == template_key or template_key in normalized_extracted or normalized_extracted in template_key:
                    count = max(count, extracted_value)
            
            template_counts[template_key] = count
        
        # Update template
        img_data['object_counts'] = template_counts
        
        # Extract present classes
        present_classes = get_all_classes_from_description(description)
        # Also include classes with count > 0
        for key, value in template_counts.items():
            if value > 0 and key not in present_classes:
                present_classes.append(key)
        
        img_data['present_classes'] = sorted(list(set(present_classes)))
        
        # Add note
        img_data['notes'] = f"Extracted from baseline VLM description. Please verify and correct counts."
        
        print(f"Processed {image_name}: found {len(present_classes)} classes, total objects: {sum(template_counts.values())}")
    
    # Save populated template
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"\nPopulated template saved to: {output_file}")
    print("Please review and correct the counts manually.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract object counts from baseline VLM descriptions and populate ground truth template'
    )
    parser.add_argument('--baseline_dir', type=str, default='./vlm_baseline_results',
                        help='Directory containing baseline descriptions')
    parser.add_argument('--template_file', type=str, default='./ground_truth_template.json',
                        help='Path to ground truth template file')
    parser.add_argument('--output_file', type=str, default='./ground_truth_annotations.json',
                        help='Output file for populated template')
    
    args = parser.parse_args()
    
    populate_template_from_baseline(
        baseline_dir=args.baseline_dir,
        template_file=args.template_file,
        output_file=args.output_file
    )


if __name__ == '__main__':
    main()
