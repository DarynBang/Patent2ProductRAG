"""
query_generation/text_captioning/openai_runner.py

This module defines a text captioning function using the OpenAI GPT-4o-mini model,
integrated into the RAG system for answering questions based on retrieved textual context.

Function:
- generate_caption_with_openai(input_data: dict) -> str:
    Formats a prompt using a shared template (`TEXT_PROMPT_TEMPLATE`) and sends it
    to OpenAI's GPT-4o-mini model for generation. Returns the model's response as a string.

Key Features:
- Uses `langchain_core.RunnableLambda` (via base_runner.py) for compatibility with LangChain.
- Supports dynamic question and context inputs.
- Logs all queries and handles API errors gracefully.
- Requires a valid `OPENAI_API_KEY` in the environment or `.env` file.

Usage:
    from query_generation.text_captioning.openai_runner import generate_caption_with_openai
    answer = generate_caption_with_openai({"query": "What is Mamba?", "texts": "..."})
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from config.prompt import PRODUCT_PROMPT_WITH_TEXT, PRODUCT_PROMPT_NO_TEXT
from utils.cluster_utils import TextClusterFilter
from typing import Tuple

# Setup logging and client
from config.logging_config import get_logger
logger = get_logger(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment or .env file.")

# === Create OpenAI Client ===
client = OpenAI(api_key=OPENAI_API_KEY)

cluster_filter = TextClusterFilter(
    model_name="all-mpnet-base-v2",
    n_clusters=3,
)

def generate_caption_with_openai(input_data: dict) -> Tuple[str, bool]:
    firm_text = input_data.get("firm_text", "")
    firm_keywords = input_data.get("firm_keywords", "")
    logger.info(f"[OpenA] Received Firm Keywords: {firm_keywords}")

    use_text = firm_text != "" and cluster_filter.is_meaningful(firm_text)
    if use_text:
        logger.info("[OpenAI] Running Product Suggestion Agent with both Text + Keywords")
        prompt = PRODUCT_PROMPT_WITH_TEXT.format(firm_text=firm_text, firm_keywords=firm_keywords)
    else:
        logger.info("[OpenAI] Running Product Suggestion Agent with Keywords only")
        prompt = PRODUCT_PROMPT_NO_TEXT.format(firm_keywords=firm_keywords)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant for document understanding and question answering."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()
        return answer, use_text

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return "OpenAI failed to generate a response.", use_text



