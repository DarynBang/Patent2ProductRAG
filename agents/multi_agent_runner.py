
# agents/multi_agent_runner.py

from typing import List, Dict, Optional, Any
from agents.base import BaseAgent
from agents.registry import AGENTS
from dotenv import load_dotenv
import logging
import gc
import pandas as pd
import torch
import ast
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentRunner:
    def __init__(self, firm_summary_rag):
        self.agents: List[BaseAgent] = []
        self.shared_memory: Dict[str, any] = {}
        self.firm_rag = firm_summary_rag
        self.query: Optional[str] = None  # to be set by PlanningAgent

    def register_agent(self, agent_name: str, qa_model: str = "qwen"):
        cls = AGENTS[agent_name]
        agent = cls(name=agent_name, qa_model=qa_model)
        self.agents.append(agent)

    def _plan_query(self) -> None:
        """Optionally run PlanningAgent to produce a refined query."""
        planning_agent = next((a for a in self.agents if a.name == "PlanningAgent"), None)
        if planning_agent:
            logger.info(f"🤖 Running {planning_agent.name}...")
            refined = planning_agent.run(self.shared_memory)
            logger.info(f"🧠 Query after planning: {refined}")
            self.shared_memory["planned_query"] = refined
            self.query = refined


    def _retrieve_firms(self, top_k: int) -> List[Dict[str, Any]]:
        """Use RAG to get top_k firm contexts for the current query."""
        with torch.inference_mode():
            results = self.firm_rag.retrieve_firm_contexts(self.query, top_k=top_k)
        logger.info(f"🏷 Retrieved {len(results)} firms")
        return results

    def _fetch_text_for_company(self,
                                company_id: int,
                                mapping: pd.DataFrame) -> str:
        """Lookup collapsed_text by hojin_id, handling missing data."""
        try:
            text = mapping.loc[mapping["hojin_id"] == company_id, "collapsed_text"].iat[0]
            if pd.isna(text):
                raise KeyError
            return text
        except Exception:
            logger.warning(f"No text found for firm {company_id}")
            return ""


    def run(self,
            initial_input: Dict[str, str],
            planning: bool = False,
            top_k: int = 5,
            firm_id_to_text_mapping: pd.DataFrame = {}):
        # Load initial inputs
        self.shared_memory.clear()
        self.shared_memory.update(initial_input)
        patent_abstract = initial_input.get("patent_abstract", "")

        # Planning phase (optional)
        if planning:
            # Find and run the PlanningAgent
            self._plan_query()

        else:
            # No planning
            self.query = patent_abstract

        # Retrieval using RAG
        rag_results = self._retrieve_firms(top_k)

        # print("Firm IDS retrieved:")
        # for company in rag_results:
        #     company_id = int(company['company_id'])
        #     company_keywords = ast.literal_eval(company['company_keywords'])
        #     text = self._fetch_text_for_company(company_id, firm_id_to_text_mapping)
        #     print(f'Company ID: {company_id}')
        #     print(f'Extracted text from company webpage(s): {text}')
        #     print(f'Company keywords: {company_keywords}')
        #
        #     if text is np.nan:
        #         print("Company ID {company_id} has no corresponding extracted text!")

        # free GPU memory if needed
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("🗑️ Freed GPU memory after retrieval")
        except Exception as e:
            logger.info(f"⚠️ Error freeing memory: {e}")

        # Per-firm Product Suggestions
        product_suggestions: Dict[int, str] = {}
        used_text_flags: Dict[int, bool] = {}
        ps_agent = next((a for a in self.agents if a.name == "ProductSuggestionAgent"), None)

        if ps_agent:
            for firm in rag_results:
                cid = int(firm["company_id"])
                company_name = firm['company_name']
                context = {
                    **self.shared_memory,
                    "firm_keywords": ast.literal_eval(firm["company_keywords"]),
                    "firm_text": self._fetch_text_for_company(cid, firm_id_to_text_mapping),
                }
                # print(f'Company ID: {cid}')
                # # print('Extracted text from company webpage(s): ' + context['firm_text'])
                # if context['firm_text'] == "":
                #     print(f"Company with ID {cid} has no Text")
                # print('Company keywords: ' + str(context['firm_keywords']))

                logger.info(f"🤖 Running {ps_agent.name} for firm {cid} - {company_name}...")
                try:
                    suggestion, used_text = ps_agent.run(input_data=context)
                    logger.info(f"{ps_agent.name}@{cid} → {suggestion}")
                except Exception as e:
                    logger.error(f"Error in {ps_agent.name}@{cid}: {e}")
                    suggestion, used_text = "", False
                product_suggestions[cid] = suggestion
                used_text_flags[cid] = used_text

        #    Downstream agents (MarketAnalysisAgent)

        # ma_agent = next((a for a in self.agents if a.name == "MarketAnalysisAgent"), None)
        # market_analysis_output: Optional[str] = None
        #
        # if ma_agent:
        #     logger.info(f"🤖 Running {ma_agent.name} on all firms...")
        #     try:
        #         market_analysis_output = ma_agent.run(
        #             input_data=self.shared_memory,
        #             firm_summary_contexts=rag_results,
        #             product_suggestions=product_suggestions
        #         )
        #         logger.info(f"{ma_agent.name} → {market_analysis_output}")
        #     except Exception as e:
        #         logger.error(f"Error in {ma_agent.name}: {e}")


        # Return the final agent's output
        return {
            "query": self.query,
            "retrieved_firms": rag_results,
            "product_suggestions": product_suggestions,
            "firm_used_text": used_text_flags,
            # "market_analysis": market_analysis_output,
        }
