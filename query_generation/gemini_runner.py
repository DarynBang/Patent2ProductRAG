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
from config.prompt import PLANNING_PROMPT_TEMPLATE

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

def generate_caption_with_gemini(input_data: dict) -> str:
    patent_abstract = input_data.get("patent_abstract", "")
    logger.info(f"[Gemini] Received query: {patent_abstract}")

    if client is None:
        return "Gemini client not initialized."

    prompt = PLANNING_PROMPT_TEMPLATE.format(patent_abstract=patent_abstract)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return "Gemini failed to generate a response."


