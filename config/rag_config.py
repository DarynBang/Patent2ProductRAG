"""
config/rag_config.py

Configuration settings for the RAG (Retrieval-Augmented Generation) system.

This module defines the firm_config dictionary that controls various aspects
of the firm data processing and retrieval system.

Configuration Parameters:
- firm_csv: Path to CSV file containing firm data with keywords and summaries
- firm_id_to_text_mapping: Path to CSV mapping firm IDs to extracted webpage text
- embed_model: Sentence transformer model for semantic embeddings
- top_k: Number of top results to retrieve during search
- output_subdir: Subdirectory for storing processed index files
- chroma_subdir: Subdirectory for ChromaDB vector database storage
- collection_name: Name of the ChromaDB collection
- retrieval_mode: Strategy for retrieval ("semantic", "keyword", "mixed")
- force_reindex: Whether to rebuild indices from scratch

Usage:
    from config.rag_config import firm_config
    pipeline = RAGPipeline(firm_config=firm_config)
"""

# EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

firm_config = {
    "firm_csv": r'firms_summary_keywords_qwen.csv',
    "firm_id_to_text_mapping": r'firm_id_to_text_mapping.csv',
    "embed_model": "sentence-transformers/all-MiniLM-L12-v2",          # or 'mpnet', 'bge'
    "top_k": 3,
    "output_subdir": r"firm_summary_index",
    "chroma_subdir": r"firm_data/chroma_db",
    "collection_name": f"firm_summary_index",
    "retrieval_mode": "mixed",
    "force_reindex": False
}
