"""
agents/product_suggestion_agent.py
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



