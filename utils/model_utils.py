"""
utils/model_utils.py

Model loading and management utilities for Qwen Vision-Language models.

This module provides lazy loading functionality for Qwen2.5-VL models,
which are used for vision-language understanding tasks in the agent system.
The models support both text and image inputs for multimodal reasoning.

Key Features:
- Singleton pattern for efficient model reuse
- 8-bit quantization for memory efficiency
- GPU optimization with proper device mapping
- Configurable model variants (3B, 7B instructions)
- BitsAndBytesConfig for quantization settings

Models Supported:
- Qwen/Qwen2.5-VL-3B-Instruct: Lightweight vision-language model
- Qwen/Qwen2.5-VL-7B-Instruct: Larger capacity model (commented)

Usage:
    model, processor = get_qwen_vl_model_and_processor()
    # Use model and processor for vision-language tasks

Note: Requires significant GPU memory (8GB+ recommended for 3B model).
The models are cached globally to avoid repeated loading overhead.

Dependencies:
    - transformers: HuggingFace model library
    - torch: PyTorch framework
    - bitsandbytes: Quantization library
"""

from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
import torch


_model: Qwen2_5_VLForConditionalGeneration = None
_processor: AutoProcessor = None

def get_qwen_vl_model_and_processor():
    """
    Lazily load and return a singleton Qwen2.5-VL model + processor.
    Subsequent calls reuse the same objects.
    """
    global _model, _processor
    if _model is None or _processor is None:
        # — load model
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        #
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_name = r"unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
        print(f"📦 Loading {model_name} model & processor…")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()

        _processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=640 * 28 * 28
        )
        print("✅ Loaded shared Qwen2.5-VL model + processor")
    return _model, _processor
