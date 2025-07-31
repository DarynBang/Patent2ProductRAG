import logging
from firm_summary_rag import FirmSummaryRAG
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
import os
import gc
import warnings
import pandas as pd

# --- new code --- 
warnings.filterwarnings("ignore")


# import torch
# torch.set_default_dtype(torch.bfloat16)

class InternshipRAG_Pipeline:
    def __init__(self, index_dir, firm_config, agent_config, ingest_only=False):
        """
        Initialize the M3APipeline.
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
            # self.multi_agent.register_agent("MarketAnalysisAgent", qa_model=qa_market_analyst)

    def ingest_firm(self, force_reindex=False) -> None:
        self.firm_rag.ingest_all(force_reindex=force_reindex)


    def add_firm_summary_to_index(self,
                                  company_id: str,
                                  company_name: str,
                                  company_keywords: str,
                                  summary_text: str
                                  ):
        self.firm_rag.add_one(company_id=company_id,
                              company_name=company_name,
                              company_keywords=company_keywords,
                              summary_text=summary_text)


    def process_query(self, patent_abstract: str, top_k: int = 5, planning=False):
        # Pass context + question into the multi-agent system
        return self.multi_agent.run({"patent_abstract": patent_abstract}, top_k=top_k, planning=planning, firm_id_to_text_mapping=self.firm_to_text_mapping_df)


def main():
    from config.agent_config import agent_config
    from config.rag_config import firm_config

    # Initialize pipeline with configs

    # Consistent index folder under PDF dir
    index_dir = r"C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\RAG_INDEX"
    patent_abstract = """An apparatus and a method for diagnosis are provided.
The apparatus for diagnosis lesion include: a model generation unit configured to categorize learning data into one or more categories and to generate
one or more categorized diagnostic models based on the categorized learning data, a model selection unit configured to select one or more diagnostic model
for diagnosing a lesion from the categorized diagnostic models, and a diagnosis unit configured to diagnose the lesion based on image data of the lesion
and the selected one or more diagnostic model.
"""

    pipeline = InternshipRAG_Pipeline(
        index_dir=index_dir,
        agent_config=agent_config,
        firm_config=firm_config,
        ingest_only=False
    )

    # Run ingestion
    pipeline.ingest_firm(force_reindex=False)

    pipeline.process_query(patent_abstract)


if __name__ == '__main__':
    main()
