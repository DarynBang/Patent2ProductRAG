"""
agents/planning_agent.py

Planning Agent for Query Optimization in the RAG System.

This module implements the PlanningAgent class, which is responsible for
optimizing user queries to improve retrieval performance. The agent takes
raw patent abstracts and reformulates them into more effective search queries
for the RAG system.

Key Features:
- Query refinement and optimization using LLMs
- Support for multiple backend models (OpenAI, Gemini, Qwen)
- Patent abstract analysis and keyword extraction
- Search query enhancement for better firm matching

The PlanningAgent uses prompt engineering to:
- Extract key technical concepts from patent abstracts
- Identify relevant business domains and applications
- Generate optimized search queries for firm retrieval
- Improve the precision of semantic search results

Usage:
    agent = PlanningAgent(name="PlanningAgent", qa_model="gemini")
    optimized_query = agent.run({"patent_abstract": "..."})

Dependencies:
    - query_generation.base_runner: Backend LLM runners
    - agents.base: Abstract base agent class
    - torch: GPU memory management
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
    """
    Agent responsible for optimizing search queries from patent abstracts.
    
    The PlanningAgent analyzes patent abstracts and generates more effective
    search queries for firm retrieval by extracting key concepts, identifying
    business applications, and formulating targeted search terms.
    
    Attributes:
        qa_model (str): Backend model for query generation
        llm: Language model runner for query optimization
    
    Args:
        name (str): Agent identifier (default: "PlanningAgent")
        qa_model (str): Backend model ("openai", "gemini", "qwen")
    """
    def __init__(self, name: str = "PlanningAgent", qa_model = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing PlanningAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.llm = get_text_captioning_runner(qa_model)

    def run(self,
            input_data: dict) -> str:
        """
        Execute query optimization on the input patent abstract.
        
        Args:
            input_data (dict): Dictionary containing:
                - patent_abstract (str): Raw patent abstract text
        
        Returns:
            str: Optimized search query for firm retrieval
            
        The method processes the patent abstract through the configured LLM
        to generate a more effective search query that better matches relevant
        firms in the database.
        """
        patent_abstract = input_data.get("patent_abstract", "")
        
        if not patent_abstract:
            logger.warning("Missing patent_abstract. Skipping Planning Agent.")
            return "No answer found."

        # Generate answer using the selected model
        return self.llm.invoke({"patent_abstract": patent_abstract})


