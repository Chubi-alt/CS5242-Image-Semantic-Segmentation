"""
VLM API-based reasoning using OpenAI GPT-4V or other API services
Supports API key access instead of local model deployment
"""

import os
import base64
from typing import Optional
from PIL import Image
import numpy as np
import io


class VLMAPIReasoner:
    """
    VLM Reasoner using API services (OpenAI GPT-4V, etc.)
    """
    
    def __init__(
        self,
        api_provider: str = "openai",
        api_key: str = None,
        model_name: str = "gpt-4o",
        max_tokens: int = 1024
    ):
        """
        Initialize API-based VLM reasoner.
        
        Args:
            api_provider: API provider ("openai", "anthropic", etc.)
            api_key: API key (or set environment variable)
            model_name: Model name (e.g., "gpt-4o", "gpt-4-vision-preview")
            max_tokens: Maximum tokens to generate
        """
        self.api_provider = api_provider.lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            if self.api_provider == "openai":
                self.api_key = os.getenv('OPENAI_API_KEY')
            elif self.api_provider == "anthropic":
                self.api_key = os.getenv('ANTHROPIC_API_KEY')
            else:
                self.api_key = None
        
        if not self.api_key:
            raise ValueError(
                f"API key not provided. Set {self.api_provider.upper()}_API_KEY environment variable "
                f"or pass api_key parameter."
            )
        
        # Initialize API client
        self._init_client()
    
    def _init_client(self):
        """Initialize API client based on provider."""
        if self.api_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        
        elif self.api_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _image_to_base64(self, image) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def reason_about_instance(
        self,
        instance_image,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Perform reasoning using API.
        
        Args:
            instance_image: Image (PIL Image, numpy array, or file path)
            prompt: Text prompt
            max_new_tokens: Maximum tokens (overrides init value if provided)
        
        Returns:
            Generated text response
        """
        max_tokens = max_new_tokens or self.max_tokens
        
        # Load image
        if isinstance(instance_image, str):
            image = Image.open(instance_image).convert('RGB')
        elif isinstance(instance_image, np.ndarray):
            image = Image.fromarray(instance_image)
        else:
            image = instance_image
        
        # Convert to base64
        img_base64 = self._image_to_base64(image)
        
        if self.api_provider == "openai":
            return self._call_openai(img_base64, prompt, max_tokens)
        elif self.api_provider == "anthropic":
            return self._call_anthropic(img_base64, prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.api_provider}")
    
    def _call_openai(self, img_base64: str, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def _call_anthropic(self, img_base64: str, prompt: str, max_tokens: int) -> str:
        """Call Anthropic Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")
    
    def reason_with_multiple_images(
        self,
        images: list,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Reason about multiple images (e.g., original + mask).
        
        Args:
            images: List of images (PIL Images, numpy arrays, or file paths)
            prompt: Text prompt
            max_new_tokens: Maximum tokens
        
        Returns:
            Generated text response
        """
        max_tokens = max_new_tokens or self.max_tokens
        
        # Convert all images to base64
        img_base64_list = [self._image_to_base64(img) for img in images]
        
        if self.api_provider == "openai":
            # OpenAI supports multiple images
            content = []
            for img_base64 in img_base64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
            content.append({
                "type": "text",
                "text": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        elif self.api_provider == "anthropic":
            # Anthropic also supports multiple images
            content = []
            for img_base64 in img_base64_list:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })
            content.append({
                "type": "text",
                "text": prompt
            })
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            return message.content[0].text.strip()
        
        else:
            raise ValueError(f"Unsupported provider: {self.api_provider}")
