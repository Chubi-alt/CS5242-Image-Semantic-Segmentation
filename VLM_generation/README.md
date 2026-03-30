# Stage 2: VLM Multimodal Alignment & Generation

This directory contains the implementation for Stage 2 of the project, which performs mask-guided instance isolation, targeted multimodal reasoning using VLM (LLaVA), and evaluation.

## Overview

The Stage 2 pipeline consists of three main components:

1. **Mask-Guided Instance Isolation**: Extract instances by multiplying the winning model's predicted masks with raw images, applying padding to filter out background noise.

2. **Targeted Multimodal Reasoning**: Pass the isolated instances into an open-source VLM (e.g., LLaVA) for targeted reasoning.

3. **Evaluation**: Qualitatively and quantitatively compare the isolated-instance outputs against a baseline using raw, unmasked images.

## Files

- `mask_guided_isolation.py`: Mask-guided instance extraction and isolation
- `vlm_reasoning.py`: VLM (LLaVA) integration for targeted reasoning
- `evaluation.py`: Qualitative and quantitative evaluation
- `main.py`: Main script that integrates all components
- `vlm_baseline.py`: **Baseline VLM script for full image descriptions** (focuses on object counts and positions)
- `requirements.txt`: Required Python packages

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

For VLM functionality, you may need additional dependencies:

```bash
# For LLaVA models
pip install transformers accelerate bitsandbytes
```

## Usage

### Single Image Processing

Process a single image with its segmentation mask:

```bash
python main.py \
    --mode single \
    --image_path ../data/test/Seq05VD_f00240.png \
    --mask_path ../UNet_baseline/test_results/Seq05VD_f00240_pred.png \
    --class_dict ../data/class_dict.csv \
    --output_dir ./results_single \
    --padding 10 \
    --min_area 100 \
    --task description
```

### Test Set Processing

Process the entire test set:

```bash
python main.py \
    --mode test_set \
    --test_images_dir ../data/test \
    --test_masks_dir ../UNet_baseline/test_results \
    --class_dict ../data/class_dict.csv \
    --output_dir ./results_test_set \
    --mask_suffix _pred.png \
    --padding 10 \
    --min_area 100 \
    --task description \
    --max_images 10
```

### Arguments

#### General Arguments
- `--mode`: Processing mode (`single` or `test_set`)
- `--image_path`: Path to input image (for single mode)
- `--mask_path`: Path to segmentation mask (for single mode)
- `--test_images_dir`: Directory containing test images (for test_set mode)
- `--test_masks_dir`: Directory containing test masks (for test_set mode)
- `--class_dict`: Path to class_dict.csv
- `--output_dir`: Output directory for results
- `--mask_suffix`: Mask filename suffix for test-set masks (default: `_pred.png`)

#### VLM Settings
- `--vlm_model`: VLM model name (default: `llava-hf/llava-1.5-7b-hf`)
- `--load_in_4bit`: Load VLM model in 4-bit quantization (saves memory)

#### Instance Extraction Settings
- `--padding`: Padding around instances in pixels (default: 10)
- `--min_area`: Minimum area for valid instances in pixels (default: 100)
- `--target_classes`: List of class indices to process (None = all classes)
- `--exclude_classes`: List of class indices to exclude (e.g., void/background)

#### Reasoning Settings
- `--task`: Type of reasoning task:
  - `description`: General description
  - `attributes`: Key attributes and characteristics
  - `context`: Context in the scene
  - `detailed`: Detailed description

#### Test Set Settings
- `--max_images`: Maximum number of images to process (None = all)

### VLM Baseline: Full Image Descriptions

The `vlm_baseline.py` script generates descriptions for full images using VLM, with a focus on object counts and relative positions. This serves as a baseline for comparison with the mask-guided approach.

#### Single Image

```bash
python vlm_baseline.py \
    --mode single \
    --image_path ../data/test/0001TP_006690.png \
    --output_dir ./vlm_baseline_results \
    --prompt_style comprehensive \
    --load_in_8bit \
    --max_memory_gb 16
```

#### Directory of Images

```bash

python vlm_baseline.py \
    --mode directory \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_baseline_results_4B \
    --prompt_style concise \
    --max_images 10 \
    --load_in_8bit \
    --max_memory_gb 16
```

#### Arguments for VLM Baseline

- `--mode`: Processing mode (`single` or `directory`)
- `--image_path`: Path to single image (for single mode)
- `--images_dir`: Directory containing images (for directory mode)
- `--output_dir`: Output directory for results
- `--vlm_model`: VLM model name (default: `llava-hf/llava-1.5-7b-hf`)
- `--load_in_4bit`: Load model in 4-bit quantization (most memory efficient)
- `--load_in_8bit`: Load model in 8-bit quantization (balanced, default)
- `--max_memory_gb`: Maximum GPU memory in GB (default: 16.0)
- `--prompt_style`: Prompt style - `detailed`, `count_focused`, `position_focused`, or `comprehensive` (default)
- `--custom_prompt`: Custom prompt (overrides prompt_style)
- `--max_new_tokens`: Maximum tokens to generate (default: 512)
- `--max_images`: Maximum number of images to process (for directory mode)

#### Prompt Styles

- **`comprehensive`** (default): Complete description with object counts, positions, and spatial relationships
- **`count_focused`**: Emphasizes counting objects and their locations
- **`position_focused`**: Emphasizes spatial layout and relative positions
- **`detailed`**: General detailed description

#### Baseline Output

The script generates:
- Individual JSON files for each image: `{image_name}_description.json`
- Combined results: `all_descriptions.json`
- Summary report: `description_report.txt`

Each description includes:
- Object counts for each type
- Relative positions (left, right, center, foreground, background)
- Spatial relationships between objects

## Output Structure

The output directory will contain:

```
output_dir/
├── isolated_instances/          # Extracted instance images
│   └── {image_name}_class{idx}_{class_name}_inst{idx}.png
├── responses/                   # VLM responses
│   ├── {image_name}_isolated.json
│   └── {image_name}_baseline.json
└── evaluation/                  # Evaluation results
    ├── qualitative_comparison.json
    ├── quantitative_results.json
    └── evaluation_report.txt
```

For test_set mode, there will also be:
```
output_dir/
├── overall_evaluation/          # Overall evaluation across all images
│   ├── qualitative_comparison.json
│   ├── quantitative_results.json
│   └── evaluation_report.txt
└── {image_name}/                 # Per-image results
    ├── isolated_instances/
    ├── responses/
    └── evaluation/
```

## Evaluation Metrics

The evaluation module provides both qualitative and quantitative metrics:

### Qualitative Metrics
- Response comparisons between isolated instances and baseline
- Word overlap analysis
- Length ratio analysis

### Quantitative Metrics
- **Length Ratio**: Ratio of response lengths
- **Word Overlap**: Overlap of words between responses
- **Specificity**: How much more specific isolated responses are
- **Relevance**: Relevance of responses to the target class

All metrics are calculated both overall and per-class.

## Workflow

1. **Load Segmentation Masks**: The system loads RGB-encoded segmentation masks from the trained UNet model.

2. **Extract Instances**: For each class in the mask:
   - Find connected components (instances)
   - Extract bounding boxes with padding
   - Isolate instances by masking out background

3. **VLM Reasoning**:
   - **Isolated Instances**: Pass each isolated instance to VLM with targeted prompts
   - **Baseline**: Pass raw images to VLM for comparison

4. **Evaluation**:
   - Compare responses qualitatively
   - Calculate quantitative metrics
   - Generate comprehensive reports

## Example

After training the UNet model and generating test results:

```bash
# Step 1: Generate segmentation masks (from UNet_baseline)
cd ../UNet_baseline
python test.py --checkpoint checkpoints/best.pth --output_dir ./test_results

# Step 2: Run VLM generation and evaluation
cd ../VLM_generation
python main.py \
    --mode test_set \
    --test_images_dir ../data/test \
    --test_masks_dir ../UNet_baseline/test_results \
    --output_dir ./vlm_results \
    --max_images 5
```

## Notes

- The VLM model (LLaVA) requires significant GPU memory. Use `--load_in_4bit` for memory-efficient loading.
- If VLM is not available, the system will use a mock VLM for testing purposes.
- Instance extraction filters out small instances based on `--min_area` threshold.
- Background classes (e.g., Void, Sky) can be excluded using `--exclude_classes`.

## Troubleshooting

### VLM Model Loading Issues
If you encounter issues loading the VLM model:
- Ensure you have sufficient GPU memory
- Try using `--load_in_4bit` for quantization
- Check that transformers library is properly installed
- Verify model name is correct

### Memory Issues
- Reduce `--max_images` for test set processing
- Use `--load_in_4bit` flag
- Process images one at a time using `--mode single`

### Mask Format Issues
- Ensure masks are RGB-encoded and match the class_dict.csv format
- Check that mask filenames match image filenames and suffix.
  UNet predictions from `UNet_baseline/test.py` use `_pred.png`.
