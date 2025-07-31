"""
query_generation/text_captioning/qwen_runner.py

This module defines the text captioning function for Qwen2.5-VL, used within the RAG system
to answer user queries based on retrieved text chunks. It formats the inputs into a prompt,
runs inference using a locally loaded Qwen2.5-VL model, and returns the generated answer.

Function:
- generate_caption_with_qwen(input_data: dict) -> str:
    Accepts a user question and associated text context, formats them using a shared prompt template,
    then runs the Qwen model to generate a concise answer.

Key Features:
- Loads the Qwen2.5-VL model and processor only once (singleton via `get_qwen_vl_model_and_processor()`).
- Formats prompts using `TEXT_PROMPT_TEMPLATE` from `config.prompt`.
- Uses Hugging Face-style `.generate()` and `.batch_decode()` to obtain the final output.
- Logs input details and supports GPU execution with memory offloading.

Usage:
    from query_generation.text_captioning.qwen_runner import generate_caption_with_qwen
    answer = generate_caption_with_qwen({"query": "What is Mamba?", "texts": "..."})
"""

import torch
import os
import logging
from typing import Tuple
from config.prompt import PRODUCT_PROMPT_WITH_TEXT, PRODUCT_PROMPT_NO_TEXT
from utils.model_utils import get_qwen_vl_model_and_processor
from utils.cluster_utils import TextClusterFilter

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Offload directory setup ===
# os.makedirs("offload", exist_ok=True)
# torch.set_default_dtype(torch.bfloat16)

cluster_filter = TextClusterFilter(
    model_name="all-mpnet-base-v2",
    n_clusters=3,
)

def generate_caption_with_qwen(input_data: dict) -> Tuple[str, bool]:
    firm_text = input_data.get("firm_text", "")
    firm_keywords = input_data.get("firm_keywords", "")
    logger.info(f"[Qwen] Received Firm Keywords: {firm_keywords}")

    # === Load Qwen2.5-VL (only once) ===
    model, processor = get_qwen_vl_model_and_processor()

    # === Begin captioning ===

    # if firm_text != "":
    #     logger.info("✅ [Qwen] Running Product Suggestion Agent with both Text + Keywords")
    #     prompt = PRODUCT_PROMPT_WITH_TEXT.format(firm_text=firm_text, firm_keywords=firm_keywords)
    #
    # else:
    #     logger.info("✅ [Qwen] Running Product Suggestion Agent with Keywords only")
    #     prompt = PRODUCT_PROMPT_NO_TEXT.format(firm_keywords=firm_keywords)

    use_text = firm_text != "" and cluster_filter.is_meaningful(firm_text)
    if use_text:
        logger.info("✅ [Qwen] Running Product Suggestion Agent with both Text + Keywords")
        prompt = PRODUCT_PROMPT_WITH_TEXT.format(firm_text=firm_text, firm_keywords=firm_keywords)
    else:
        logger.info("✅ [Qwen] Running Product Suggestion Agent with Keywords only")
        prompt = PRODUCT_PROMPT_NO_TEXT.format(firm_keywords=firm_keywords)

    messages = [
        {
            "role": "user", 
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # print("\n--- Inputs to Textual Model.generate() ---")
    # for k, v in inputs.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
    #     else:
    #         print(f"{k}: type={type(v)}, value={v}")
    # print("----------------------------------\n")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    inputs.to("cpu")
    del generated_ids, generated_ids_trimmed, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text[0], use_text

