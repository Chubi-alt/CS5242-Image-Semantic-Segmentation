# VLM API 使用指南

使用 API key 访问 VLM 模型，无需本地部署。

## 支持的 API 提供商

- **OpenAI**: GPT-4o, GPT-4 Vision, GPT-4 Turbo with Vision
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku

## 安装依赖

```bash
# OpenAI
pip install openai

# Anthropic (可选)
pip install anthropic
```

## 设置 API Key

### 方法 1: 环境变量（推荐）

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 方法 2: 命令行参数

在运行命令时使用 `--api_key` 参数。

## 使用方法

### 1. Baseline 描述（仅原始图片）

```bash
cd /data/lingyu.li/CS5242-Image-Semantic-Segmentation/VLM_generation

python vlm_baseline_api.py \
    --mode directory \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_baseline_results \
    --api_provider openai \
    --model_name gpt-4o \
    --prompt_style concise \
    --max_new_tokens 1024
```

### 2. Mask 描述（原始图片 + 分割掩码）

```bash
python vlm_with_mask_api.py \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_mask_results \
    --mask_dir ../UNet_baseline/test_results \
    --api_provider openai \
    --model_name gpt-4o \
    --prompt_style concise \
    --max_new_tokens 1024
```

## 参数说明

### API 设置
- `--api_provider`: API 提供商 (`openai` 或 `anthropic`)
- `--api_key`: API key（可选，优先使用环境变量）
- `--model_name`: 模型名称
  - OpenAI: `gpt-4o`, `gpt-4-vision-preview`, `gpt-4-turbo`
  - Anthropic: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

### 其他参数
- `--prompt_style`: Prompt 风格 (`concise`, `detailed`, `comprehensive` 等)
- `--max_new_tokens`: 最大生成 token 数
- `--max_images`: 处理的最大图片数量（可选）

## 使用 Anthropic Claude

```bash
python vlm_baseline_api.py \
    --mode directory \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_baseline_results \
    --api_provider anthropic \
    --model_name claude-3-opus-20240229 \
    --prompt_style concise
```

## 优势

1. **无需 GPU**: 不需要本地 GPU 资源
2. **无需下载模型**: 不需要下载大型模型文件
3. **更快的启动**: 无需加载模型，直接调用 API
4. **更好的性能**: GPT-4o 和 Claude 3 通常比本地小模型性能更好
5. **多图片支持**: API 原生支持多图片输入（原始图片 + 掩码）

## 成本考虑

- **OpenAI GPT-4o**: 按 token 计费，图片按分辨率计费
- **Anthropic Claude**: 按 token 计费，图片按分辨率计费

建议：
- 小规模测试：使用 API 很方便
- 大规模处理：考虑成本，可能需要本地部署

## 示例：完整流程

```bash
# 1. 设置 API key
export OPENAI_API_KEY="sk-..."

# 2. 生成 baseline 描述
python vlm_baseline_api.py \
    --mode directory \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_baseline_results \
    --api_provider openai \
    --model_name gpt-4o \
    --prompt_style concise

# 3. 生成 mask 描述
python vlm_with_mask_api.py \
    --images_dir ../data/vlm_attempt \
    --output_dir ./vlm_mask_results \
    --mask_dir ../UNet_baseline/test_results \
    --api_provider openai \
    --model_name gpt-4o \
    --prompt_style concise

# 4. 评估结果
python evaluate_vlm_descriptions.py \
    --baseline_dir ./vlm_baseline_results \
    --mask_results_dir ./vlm_mask_results \
    --mask_dir ../UNet_baseline/test_results \
    --images_dir ../data/vlm_attempt \
    --ground_truth_file ./ground_truth_annotations.json \
    --class_dict ../data/class_dict.csv \
    --output ./vlm_evaluation_results.json
```
