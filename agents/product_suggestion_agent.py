"""
agents/product_suggestion_agent.py

Product Suggestion Agent for generating business recommendations.

This module implements the ProductSuggestionAgent class, which analyzes firm
information and generates relevant product suggestions based on patent abstracts
and firm capabilities. The agent uses both firm keywords and extracted webpage
text to provide targeted business recommendations.

Key Features:
- Multi-modal input processing (keywords + webpage text)
- Intelligent content filtering using text meaningfulness detection
- Support for multiple backend LLMs (OpenAI, Gemini, Qwen)
- Dual-mode operation (text+keywords vs keywords-only)
- Product recommendation generation for patent-firm matching

The agent operates in two modes:
1. Full Mode: Uses both firm keywords and meaningful webpage text
2. Keywords Mode: Falls back to keywords-only when text is not meaningful

Usage:
    agent = ProductSuggestionAgent(
        name="ProductSuggestionAgent", 
        qa_model="gemini"
    )
    suggestions, used_text = agent.run({
        "firm_text": "Company description...",
        "firm_keywords": ["AI", "machine learning", "healthcare"],
        "patent_abstract": "Medical diagnosis system..."
    })

Returns:
    Tuple[str, bool]: Product suggestions and flag indicating if text was used

Dependencies:
    - product_suggestion.base_runner: Backend LLM runners
    - agents.base: Abstract base agent class
    - torch: GPU memory management
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Tuple
from product_suggestion.base_runner import get_text_captioning_runner
import torch
import ast


class ProductSuggestionAgent(BaseAgent):
    def __init__(self, name: str = "ProductSuggestionAgent", qa_model="qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing ProductSuggestionAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.llm = get_text_captioning_runner(qa_model)

    def run(self,
            input_data: dict) -> Tuple[str, bool]:
        firm_text = input_data.get("firm_text", "")
        firm_keywords = input_data.get("firm_keywords", "")

        if firm_text == "" and not firm_keywords:
            logger.warning("Missing both extracted text and keywords. Skipping Product Suggestion Agent.")
            return "No answer found.", False

        # Generate answer using the selected model
        return self.llm.invoke({"firm_text": firm_text, "firm_keywords": firm_keywords})



