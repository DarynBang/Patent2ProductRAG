"""
agents/planning_agent.py
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from query_generation.base_runner import get_text_captioning_runner
import torch
import gc

class PlanningAgent(BaseAgent):
    def __init__(self, name: str = "PlanningAgent", qa_model = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing PlanningAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.llm = get_text_captioning_runner(qa_model)

    def run(self,
            input_data: dict) -> str:
        patent_abstract = input_data.get("patent_abstract", "")
        
        if not patent_abstract:
            logger.warning("Missing patent_abstract. Skipping Planning Agent.")
            return "No answer found."

        # Generate answer using the selected model
        return self.llm.invoke({"patent_abstract": patent_abstract})


