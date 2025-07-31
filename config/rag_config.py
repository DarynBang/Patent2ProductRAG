# EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

firm_config = {
    "firm_csv": r'C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\firms_summary_keywords_qwen.csv',
    "firm_id_to_text_mapping": r'C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\firm_id_to_text_mapping.csv',
    "embed_model": "sentence-transformers/all-MiniLM-L12-v2",          # or 'mpnet', 'bge'
    "top_k": 3,
    "output_subdir": r"firm_summary_index",
    "chroma_subdir": r"firm_data/chroma_db",
    "collection_name": f"firm_summary_index",
    "retrieval_mode": "mixed",
    "force_reindex": False
}
