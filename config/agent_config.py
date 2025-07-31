"""
config/agent_config.py

Configuration settings for the multi-agent system.

This module defines the agent_config dictionary that specifies which
language models to use for different agents in the system.

Supported Models:
- "openai": Uses OpenAI GPT models (requires OPENAI_API_KEY)
- "gemini": Uses Google Gemini models (requires GENAI_API_KEY)  
- "qwen": Uses Qwen vision-language models (local inference)

Agent Types:
- qa_planning_agent: Handles query optimization and planning
- qa_product_suggestion_agent: Generates product recommendations
- qa_market_analyst_agent: Provides market analysis and opportunities

Usage:
    from config.agent_config import agent_config
    pipeline = MultiAgentRunner(agent_config=agent_config)

Note: Ensure appropriate API keys are set in environment variables
or .env file for cloud-based models (OpenAI, Gemini).
"""

# config/agent_config.py

agent_config = {
    # Shared QA model name for all agents - text (currently support: "qwen", "gemini", "openai")
    "qa_planning_agent": "gemini",
    "qa_product_suggestion_agent": "gemini",
    "qa_market_analyst_agent": "gemini",
}


