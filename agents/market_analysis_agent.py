"""
agents/market_analysis_agent.py

This module defines the `MergeAgent`, responsible for synthesizing multiple generalized sub-answers
into a single coherent summary. It leverages a language model (e.g., GPT-4o or Qwen) to produce
a natural and non-redundant final response.

Key Features:
- Maintains internal memory of previously seen sub-answers across multiple runs.
- Uses `MERGE_PROMPT` to guide the LLM in combining multiple answer strings fluently.
- Supports both OpenAI and HuggingFace-based LLMs via LangChain pipelines.

Typical Usage:
    agent = MergeAgent(qa_model="openai")
    final_summary = agent.run({"generalized_answers": list_of_answers})
"""

import logging
from config.logging_config import get_logger

logger = get_logger(__name__)

from agents.base import BaseAgent
from config.prompt import MARKET_ANALYST_PROMPT_TEMPLATE
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline
import pprint
from typing import Optional, List
import gc
import torch

class MarketAnalysisAgent(BaseAgent):
    def __init__(self, name="MarketAnalysisAgent", qa_model="openai", ):
        super().__init__(name)
        logger.info(f"Initializing Market Analysis Agent with model: {qa_model}")

        # Instantiate the raw LLM (no prompt bound yet)
        if qa_model == "openai":
            import os

            if "OPENAI_API_KEY" not in os.environ:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("Loaded environment variables for OpenAI credentials")
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        elif "gemini" in qa_model:
            import os

            if "GENAI_API_KEY" not in os.environ:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("Loaded .env for Gemini credentials")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            self.llm = HuggingFacePipeline(
                pipeline=pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=processor.tokenizer,
                    device_map="auto",
                    return_full_text=False,  # <-- key change
                    max_new_tokens=1024,  # or whatever you need
                    clean_up_tokenization_spaces=True
                )
            )
        else:
            raise ValueError(f"Unknown qa_model: {qa_model!r}")

        # Load both templates
        self.prompt = PromptTemplate.from_template(
            MARKET_ANALYST_PROMPT_TEMPLATE
        )
        # Parser for the output
        self.parser = StrOutputParser()

    def run(self,
            input_data: dict,
            firm_summary_contexts: Optional[List[dict]] = None,
            product_suggestions: Optional[dict] = None
            ) -> str:
        patent_abstract = input_data.get("patent_abstract")
        logger.info("Running Market Analyst Agent")

        firm_chunks = [c["chunk"] for c in firm_summary_contexts]
        firm_context_str = "\n ### \n- ".join(firm_chunks)
        
        # Enhanced context with product suggestions if available
        enhanced_context = firm_context_str
        if product_suggestions:
            logger.info(f"Including product suggestions for {len(product_suggestions)} firms")
            product_context_parts = []
            for firm in firm_summary_contexts:
                company_id = int(firm.get("company_id", 0))
                if company_id in product_suggestions and product_suggestions[company_id]:
                    company_name = firm.get("company_name", f"Company {company_id}")
                    products = product_suggestions[company_id]
                    product_context_parts.append(f"{company_name} Products: {products}")
            
            if product_context_parts:
                enhanced_context += "\n\n### Product Information:\n" + "\n".join(product_context_parts)

        # Free memory before LLM call
        del firm_summary_contexts,
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        template = self.prompt
        prompt_input = {
            "firm_summary_contexts": enhanced_context,
            "patent_abstract": patent_abstract
        }

        prompt_text = template.format(**prompt_input)
        # Render and invoke the LLM
        try:
            if isinstance(self.llm, HuggingFacePipeline):
                llm_output = self.llm(prompt_text)
                final = self.parser.parse(llm_output)
            else:
                messages = [
                    HumanMessage(content=prompt_text)
                ]
                llm_response = self.llm(messages)
                final = self.parser.parse(llm_response.content)

            return final

        except Exception as e:
            logger.error(f"Market Analyst Agent failed: {str(e)}", exc_info=e)
            return "Failed to generate market opportunities."


