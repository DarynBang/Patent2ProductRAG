"""
InternshipRAG_pipeline.py

Main RAG Pipeline for Patent-to-Product Matching System

This module provides the core RAG (Retrieval-Augmented Generation) pipeline that connects
patent abstracts to relevant firms and generates product suggestions using a multi-agent
system approach.

Key Components:
- InternshipRAG_Pipeline: Main pipeline class orchestrating the RAG process
- FirmSummaryRAG: Handles firm data indexing and retrieval
- MultiAgentRunner: Manages planning, product suggestion, and market analysis agents

Features:
- Semantic search with multiple embedding models
- Multi-agent processing for query optimization and product suggestions
- Firm data ingestion and indexing with ChromaDB
- GPU memory management and optimization
- Configurable retrieval strategies (semantic, keyword, mixed)

Usage:
    pipeline = InternshipRAG_Pipeline(
        index_dir="RAG_INDEX",
        agent_config=agent_config,
        firm_config=firm_config
    )
    results = pipeline.process_query("patent abstract text", top_k=5)

Dependencies:
    - firm_summary_rag: Core RAG functionality
    - agents.multi_agent_runner: Multi-agent orchestration
    - torch: GPU memory management
    - pandas: Data handling
"""

import logging
from firm_summary_rag import FirmSummaryRAG
import torch

from config.logging_config import get_logger

logger = get_logger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
import os
import gc
import warnings
import pandas as pd
from config.rag_config import firm_config
from config.agent_config import agent_config

# --- new code --- 
warnings.filterwarnings("ignore")


# import torch
# torch.set_default_dtype(torch.bfloat16)

class InternshipRAG_Pipeline:
    """
    Main RAG Pipeline for Patent-to-Product Matching.
    
    This class orchestrates the entire pipeline from patent abstract input to
    firm recommendations and product suggestions. It integrates semantic search,
    multi-agent processing, and result generation.
    
    Attributes:
        index_dir (str): Directory for storing index files
        multi_agent (MultiAgentRunner): Agent orchestration system
        firm_to_text_mapping_df (pd.DataFrame): Mapping between firm IDs and extracted text
        firm_rag (FirmSummaryRAG): RAG system for firm data retrieval
    
    Args:
        index_dir (str): Path to directory for index storage
        firm_config (dict): Configuration for firm data and retrieval
        agent_config (dict): Configuration for multi-agent system
        ingest_only (bool): If True, only perform data ingestion without agent setup
    """
    def __init__(self, index_dir, firm_config, agent_config, ingest_only=False):
        """
        Initialize the RAG Pipeline with configuration and data setup.
        
        Sets up the vector database, loads firm data, and initializes
        the multi-agent system for query processing and product suggestions.
        """
        # Initialize RAG indexer and multi-agent QA system
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        self.multi_agent = None

        firm_df = pd.read_csv(firm_config.get("firm_csv"))
        firm_to_text_mapping = firm_config.get("firm_id_to_text_mapping", "")

        self.firm_to_text_mapping_df = pd.read_csv(firm_to_text_mapping)

        # Initialize RAG
        self.firm_rag = FirmSummaryRAG(df=firm_df,
                                       index_dir=index_dir,
                                       config=firm_config)

        if ingest_only is False:
            # one global qa_model for all agents (you could customize per‐agent too)
            qa_planning = agent_config.get("qa_planning_agent", "qwen")
            qa_product_suggestion = agent_config.get("qa_product_suggestion_agent", "gemini")
            qa_market_analyst = agent_config.get("qa_market_analyst_agent", "gemini")

            self.multi_agent = MultiAgentRunner(firm_summary_rag=self.firm_rag)

            self.multi_agent.register_agent("PlanningAgent", qa_model=qa_planning)
            self.multi_agent.register_agent("ProductSuggestionAgent", qa_model=qa_product_suggestion)
            self.multi_agent.register_agent("MarketAnalysisAgent", qa_model=qa_market_analyst)

    def ingest_firm(self, force_reindex=False) -> None:
        self.firm_rag.ingest_all(force_reindex=force_reindex)


    def add_firm_summary_to_index(self,
                                  company_id: str,
                                  company_name: str,
                                  company_keywords: str,
                                  summary_text: str
                                  ):
        """Add a new firm or update an existing one"""
        self.firm_rag.add_one(
            company_id=company_id,
            company_name=company_name,
            company_keywords=company_keywords,
            summary_text=summary_text
        )

    def process_query(self, patent_abstract: str, top_k: int = 5, planning=False):
        # Pass context + question into the multi-agent system
        return self.multi_agent.run({"patent_abstract": patent_abstract}, top_k=top_k, planning=planning, firm_id_to_text_mapping=self.firm_to_text_mapping_df)


def main():
    """
    Main function for testing the pipeline directly.
    
    This function demonstrates basic usage of the RAG pipeline
    with a sample patent abstract for diagnostic apparatus.
    """
    index_dir = r"RAG_INDEX"
    
    patent_abstract = """An apparatus and a method for diagnosis are provided.
The apparatus for diagnosis lesion include: a model generation unit configured to categorize learning data into one or more categories and to generate
one or more categorized diagnostic models based on the categorized learning data, a model selection unit configured to select one or more diagnostic model
for diagnosing a lesion from the categorized diagnostic models, and a diagnosis unit configured to diagnose the lesion based on image data of the lesion
and the selected one or more diagnostic model.
"""

    print("🚀 Initializing InternshipRAG Pipeline...")
    
    pipeline = InternshipRAG_Pipeline(
        index_dir=index_dir,
        agent_config=agent_config,
        firm_config=firm_config,
        ingest_only=False
    )

    print("📥 Running data ingestion...")
    # Run ingestion
    pipeline.ingest_firm(force_reindex=False)

    print("🔄 Processing patent abstract...")
    results = pipeline.process_query(patent_abstract)
    
    print("\n📊 Results:")
    print(f"Found {len(results.get('retrieved_firms', []))} relevant firms")
    
    if 'market_analysis' in results:
        print("✅ Market analysis included")
    else:
        print("⚠️ Market analysis not available")


if __name__ == '__main__':
    main()
