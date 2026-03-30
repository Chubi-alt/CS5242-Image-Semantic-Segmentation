"""
Targeted Multimodal Reasoning using VLM (LLaVA)
Pass isolated instances into an open-source VLM for targeted reasoning.
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Optional, Dict


class VLMReasoner:
    """
    Wrapper for VLM (LLaVA) for targeted reasoning on isolated instances.
    """
    
    def __init__(
        self,
        model_name: str = "./Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_memory_gb: float = 16.0,
        use_single_gpu: bool = True
    ):
        """
        Initialize VLM model.
        
        Args:
            model_name: Name of the LLaVA model to use
            device: Device to run on ('cuda' or 'cpu'), None for auto
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization (more stable than 4-bit)
            max_memory_gb: Maximum GPU memory to use in GB (default: 16GB)
            use_single_gpu: Force using single GPU (default: True)
        """
        self.model_name = model_name
        # Force single GPU if requested
        if use_single_gpu and torch.cuda.is_available():
            self.device = device or 'cuda:0'
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_memory_gb = max_memory_gb
        self.use_single_gpu = use_single_gpu
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load VLM model (Qwen3-VL or LLaVA) and processor with memory optimization."""
        try:
            from transformers import (
                AutoConfig,
                AutoProcessor,
                LlavaForConditionalGeneration,
                LlavaNextForConditionalGeneration,
            )
            
            # Try to import Qwen3-VL specific classes
            try:
                from transformers import Qwen3VLForConditionalGeneration
                qwen3vl_available = True
            except ImportError:
                qwen3vl_available = False
                # Fallback: try Qwen2VL
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    qwen2vl_available = True
                except ImportError:
                    qwen2vl_available = False
            
            print(f"Loading VLM model: {self.model_name}")
            print(f"Device: {self.device}")
            if self.load_in_4bit:
                print("Using 4-bit quantization")
            elif self.load_in_8bit:
                print("Using 8-bit quantization")
            if self.use_single_gpu and 'cuda' in str(self.device):
                print(f"Using single GPU with max memory: {self.max_memory_gb}GB")

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Determine model type
            model_type = config.model_type if hasattr(config, 'model_type') else None
            model_name_lower = self.model_name.lower()
            
            # Check if it's Qwen3-VL model
            is_qwen3 = ('qwen3' in model_name_lower or model_type == 'qwen3_vl')
            is_qwen2 = ('qwen2' in model_name_lower or model_type == 'qwen2_vl')
            is_qwen = is_qwen3 or is_qwen2 or ('qwen' in model_name_lower and 'vl' in model_name_lower)
            
            if is_qwen3 and qwen3vl_available:
                print("Detected Qwen3-VL model - using Qwen3VLForConditionalGeneration")
                model_cls = Qwen3VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            elif is_qwen2 and qwen2vl_available:
                print("Detected Qwen2-VL model - using Qwen2VLForConditionalGeneration")
                model_cls = Qwen2VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            elif is_qwen:
                # Try Qwen3-VL first, then Qwen2-VL
                if qwen3vl_available:
                    try:
                        print("Trying Qwen3VLForConditionalGeneration...")
                        model_cls = Qwen3VLForConditionalGeneration
                        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                        print("Successfully loaded as Qwen3-VL model")
                    except Exception as e:
                        if qwen2vl_available:
                            print(f"Qwen3-VL failed ({e}), trying Qwen2-VL...")
                            model_cls = Qwen2VLForConditionalGeneration
                            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                            print("Successfully loaded as Qwen2-VL model")
                        else:
                            raise
                elif qwen2vl_available:
                    print("Using Qwen2VLForConditionalGeneration")
                    model_cls = Qwen2VLForConditionalGeneration
                    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                else:
                    raise ValueError("Qwen3-VL or Qwen2-VL classes not available. Please install latest transformers.")
            elif model_type == 'llava':
                print("Detected LLaVA model")
                model_cls = LlavaForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            elif model_type == 'llava_next':
                print("Detected LLaVA-Next model")
                model_cls = LlavaNextForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            else:
                raise ValueError(
                    f"Unsupported VLM model type: {model_type}. "
                    "Please use a Qwen3-VL, Qwen2-VL, LLaVA or LLaVA-Next checkpoint."
                )

            # Prepare device map for single GPU
            # For Qwen3-VL/Qwen2-VL, use device_map="auto" or explicit mapping
            if is_qwen3 or (is_qwen and qwen2vl_available):
                # Qwen3-VL/Qwen2-VL: use device_map="auto" or explicit device mapping
                if self.use_single_gpu and 'cuda' in str(self.device):
                    # Use "auto" for Qwen models, it handles device mapping well
                    device_map = "auto"
                elif 'cuda' in str(self.device):
                    device_map = "auto"
                else:
                    device_map = None
            else:
                # LLaVA models
                if self.use_single_gpu and 'cuda' in str(self.device):
                    gpu_id = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0
                    device_map = {f"cuda:{gpu_id}": f"{self.max_memory_gb}GB"}
                elif 'cuda' in str(self.device):
                    device_map = "auto"
                else:
                    device_map = None

            # Load with quantization if requested
            load_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True
            }
            
            # Add device_map if specified
            if device_map is not None:
                load_kwargs['device_map'] = device_map
            
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_kwargs.update({
                    'quantization_config': quantization_config,
                    'torch_dtype': torch.float16
                })
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs.update({
                    'quantization_config': quantization_config,
                    'torch_dtype': torch.float16
                })
            else:
                # Load in float16 for memory efficiency
                load_kwargs['torch_dtype'] = torch.float16 if 'cuda' in str(self.device) else torch.float32
            
            self.model = model_cls.from_pretrained(self.model_name, **load_kwargs)
            
            # For Qwen3-VL with device_map="auto", model should already be on correct device
            # But we verify and move if needed
            if device_map is None:
                # If no device_map, manually move to device
                print(f"Moving model to {self.device}...")
                self.model = self.model.to(self.device)
            else:
                # If using device_map, verify it worked correctly
                try:
                    first_param = next(self.model.parameters())
                    actual_device = first_param.device
                    print(f"Model device (from device_map): {actual_device}")
                    # For Qwen3-VL, device_map="auto" should handle it correctly
                    # But if model is on CPU when CUDA was requested, move it
                    if 'cuda' in str(self.device) and 'cpu' in str(actual_device):
                        print(f"Warning: Model on {actual_device}, expected CUDA. Moving to {self.device}...")
                        self.model = self.model.to(self.device)
                except Exception as e:
                    print(f"Warning: Could not verify model device: {e}")
                    # Fallback: try to move to device
                    try:
                        self.model = self.model.to(self.device)
                    except:
                        pass
            
            # Enable evaluation mode and disable gradient computation
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Clear cache
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            
            print("VLM model loaded successfully!")
            
        except ImportError as e:
            print("=" * 80)
            print("ERROR: transformers library not found. VLM model cannot be loaded.")
            print("=" * 80)
            print("Install required packages with:")
            print("  pip install transformers accelerate bitsandbytes")
            print("\nFalling back to mock VLM mode. All responses will be placeholders.")
            print("=" * 80)
            self.model = None
            self.processor = None
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed to load VLM model: {e}")
            print("=" * 80)
            print("Possible causes:")
            print("  1. Insufficient GPU memory - try using --load_in_4bit or --load_in_8bit")
            print("  2. Network issues downloading the model")
            print("  3. Model name incorrect or model not accessible")
            print("  4. Missing dependencies - ensure transformers, accelerate, bitsandbytes are installed")
            print("\nFalling back to mock VLM mode. All responses will be placeholders.")
            print("=" * 80)
            self.model = None
            self.processor = None
    
    def reason_about_instance(
        self,
        instance_image: np.ndarray,
        prompt: str = "What is this object? Describe it in detail.",
        max_new_tokens: int = 256
    ) -> str:
        """
        Perform targeted reasoning on an isolated instance.
        
        Args:
            instance_image: Isolated instance image, shape [H, W, 3], uint8
            prompt: Text prompt for the VLM
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            return self._mock_reasoning(instance_image, prompt)
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(instance_image, np.ndarray):
                pil_image = Image.fromarray(instance_image)
            else:
                pil_image = instance_image

            # Check if it's Qwen3-VL model by checking processor type
            processor_type = str(type(self.processor)).lower()
            is_qwen3 = 'qwen' in processor_type
            
            if is_qwen3:
                # Qwen3-VL format - try the standard Qwen2VL API
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    # Use processor's apply_chat_template if available
                    if hasattr(self.processor, 'apply_chat_template'):
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        # Process queries for images
                        if hasattr(self.processor, 'process_queries'):
                            image_inputs, video_inputs = self.processor.process_queries(
                                messages, self.model.config
                            )
                            inputs = self.processor(
                                text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                            )
                        else:
                            # Fallback: direct processing
                            inputs = self.processor(
                                text=[text],
                                images=[pil_image],
                                padding=True,
                                return_tensors="pt",
                            )
                    else:
                        # Direct processing without chat template
                        inputs = self.processor(
                            text=[prompt],
                            images=[pil_image],
                            padding=True,
                            return_tensors="pt",
                        )
                except Exception as e:
                    print(f"Warning: Qwen3-VL specific processing failed: {e}")
                    print("Falling back to standard processing...")
                    # Fallback to standard processing
                    inputs = self.processor(
                        text=[prompt],
                        images=[pil_image],
                        padding=True,
                        return_tensors="pt",
                    )
            else:
                # LLaVA format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                prompt_text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    images=pil_image,
                    text=prompt_text,
                    return_tensors="pt",
                )
            
            # Move inputs to device - ensure they match model's device
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}, Moving inputs to {model_device}")
            
            inputs_on_device = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_device[key] = value.to(model_device)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # Handle lists/tuples that might contain tensors
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
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
                # Decode response - handle both Qwen3-VL and LLaVA formats
                if is_qwen3:
                    # For Qwen3-VL, decode the full output and extract generated part
                    if hasattr(self.processor, 'batch_decode'):
                        generated_text = self.processor.batch_decode(
                            output,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        # Remove input prompt if present
                        if prompt in generated_text:
                            generated_text = generated_text.split(prompt, 1)[-1].strip()
                    else:
                        generated_text = self.processor.decode(
                            output[0],
                            skip_special_tokens=True
                        )
                else:
                    # LLaVA format
                    generated_text = self.processor.decode(
                        output[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )

            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during VLM reasoning: {e}")
            return self._mock_reasoning(instance_image, prompt)
    
    def _mock_reasoning(self, instance_image: np.ndarray, prompt: str) -> str:
        """Mock reasoning function when VLM is not available."""
        return (
            "[MOCK VLM RESPONSE - VLM model not loaded]\n\n"
            "The VLM model failed to load. This is a placeholder response.\n"
            "To get real VLM descriptions, please:\n"
            "1. Install required packages: pip install transformers accelerate bitsandbytes\n"
            "2. Ensure you have sufficient GPU memory (16GB+ recommended)\n"
            "3. Check that the model name is correct and accessible\n\n"
            f"Prompt requested: {prompt}\n"
            f"Image shape: {instance_image.shape}\n\n"
            "NOTE: This is NOT a real VLM description. Please fix the model loading issue to get actual descriptions."
        )
    
    def batch_reason(
        self,
        instance_images: List[np.ndarray],
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 256
    ) -> List[str]:
        """
        Perform reasoning on multiple instances.
        
        Args:
            instance_images: List of isolated instance images
            prompts: List of prompts (or single prompt for all)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of generated responses
        """
        if prompts is None:
            prompts = ["What is this object? Describe it in detail."] * len(instance_images)
        elif isinstance(prompts, str):
            prompts = [prompts] * len(instance_images)
        
        responses = []
        for img, prompt in zip(instance_images, prompts):
            response = self.reason_about_instance(img, prompt, max_new_tokens)
            responses.append(response)
        
        return responses


def create_targeted_prompts(class_name: str, task: str = "description") -> str:
    """
    Create targeted prompts for specific classes and tasks.
    
    Args:
        class_name: Name of the class (e.g., "Car", "Pedestrian")
        task: Type of reasoning task ("description", "attributes", "context")
    
    Returns:
        Formatted prompt string
    """
    prompts = {
        "description": f"Describe this {class_name.lower()} in detail. What are its key visual features?",
        "attributes": f"What are the key attributes and characteristics of this {class_name.lower()}?",
        "context": f"Describe this {class_name.lower()} and its context in the scene. What is it doing?",
        "detailed": f"Provide a detailed description of this {class_name.lower()}. Include its appearance, position, and any notable features."
    }
    
    return prompts.get(task, prompts["description"])


def reason_about_all_instances(
    instances: dict,
    vlm_reasoner: VLMReasoner,
    class_dict: dict,
    task: str = "description",
    custom_prompts: Optional[Dict[int, str]] = None
) -> dict:
    """
    Perform reasoning on all isolated instances.
    
    Args:
        instances: Dictionary mapping class_idx -> list of (instance, bbox) tuples
        vlm_reasoner: VLMReasoner instance
        class_dict: Dictionary mapping class_idx -> class_name
        task: Type of reasoning task
        custom_prompts: Custom prompts for specific class indices
    
    Returns:
        Dictionary mapping (class_idx, instance_idx) -> response string
    """
    results = {}
    
    for class_idx, instance_list in instances.items():
        class_name = class_dict.get(class_idx, f"class_{class_idx}")
        
        for inst_idx, (instance, bbox) in enumerate(instance_list):
            # Get prompt
            if custom_prompts and class_idx in custom_prompts:
                prompt = custom_prompts[class_idx]
            else:
                prompt = create_targeted_prompts(class_name, task)
            
            # Perform reasoning
            response = vlm_reasoner.reason_about_instance(instance, prompt)
            
            results[(class_idx, inst_idx)] = {
                'response': response,
                'bbox': bbox,
                'class_name': class_name
            }
    
    return results
