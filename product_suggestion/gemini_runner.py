"""
query_generation/text_captioning/gemini_runner.py

This module defines a text captioning function using the Gemini 2.0 Flash model
from Google Generative AI. It is used as one of the backends in the RAG pipeline
for generating answers based on retrieved text chunks.

Function:
- generate_caption_with_gemini(input_data: dict) -> str:
    Accepts a user query and a string of retrieved context.
    Formats them using a shared prompt template (`TEXT_PROMPT_TEMPLATE`)
    and sends the prompt to Gemini 2.0 Flash for content generation.

Key Features:
- Loads Gemini API client once and reuses it for efficiency.
- Uses the prompt template defined in `config.prompt`.
- Returns a clean textual answer or error fallback.

Usage:
    from query_generation.text_captioning.gemini_runner import generate_caption_with_gemini
    answer = generate_caption_with_gemini({"query": "What is Mamba?", "texts": "..."})
"""

import logging
from google import genai
from dotenv import load_dotenv
from config.prompt import PRODUCT_PROMPT_WITH_TEXT, PRODUCT_PROMPT_NO_TEXT
from utils.cluster_utils import TextClusterFilter
from typing import Tuple

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from sentence_transformers import SentenceTransformer

# === Setup ===
load_dotenv()
from config.logging_config import get_logger
logger = get_logger(__name__)

try:
    client = genai.Client()
    logger.info("Gemini client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    client = None


cluster_filter = TextClusterFilter(
    model_name="all-mpnet-base-v2",
    n_clusters=3,
)


def generate_caption_with_gemini(input_data: dict) -> Tuple[str, bool]:
    firm_text = input_data.get("firm_text", "")
    firm_keywords = input_data.get("firm_keywords", "")
    logger.info(f"[Gemini] Received Firm Keywords and Text")

    used_text = False  # Default

    if client is None:
        return "Gemini client not initialized.", False

    if firm_text != "" and cluster_filter.is_meaningful(firm_text):
        logger.info("Text is meaningful - using TEXT+KEYWORDS prompt")
        prompt = PRODUCT_PROMPT_WITH_TEXT.format(
            firm_text=firm_text,
            firm_keywords=firm_keywords
        )
        used_text = True
    else:
        if firm_text != "":
            logger.info("Text deemed not meaningful - using KEYWORDS-only")
        else:
            logger.info("No text provided - using KEYWORDS-only")
        prompt = PRODUCT_PROMPT_NO_TEXT.format(
            firm_keywords=firm_keywords
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        return response.text.strip(), used_text
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return "Gemini failed to generate a response.", used_text


