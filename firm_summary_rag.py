import os
import json
from typing import List, Dict, Any
from tqdm.auto import tqdm
import chromadb
import numpy as np
import chromadb.utils.embedding_functions as embedding_functions
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from config.rag_config import firm_config
from typing import Optional
from rank_bm25 import BM25Okapi
import nltk
import pickle

# nltk.download('punkt_tab')    <- Ensure punkt_tab is downloaded for

class FirmSummaryRAG:
    def __init__(
        self,
        df,
        index_dir: str,
        config: dict
    ):
        # process patent ->
        # Data + paths
        self.df = df
        self.index_dir = index_dir
        self.embed_model = config.get("embed_model")
        self.device = config.get("device", torch.cuda.is_available() and "cuda" or "cpu")
        self.chunk_size = config.get("chunk_size", 2048)
        self.chunk_overlap = config.get("chunk_overlap", 256)
        self.batch_size = config.get("batch_size", 5000)
        self.top_k = config.get("top_k", 5)
        self.output_subdir = config.get("output_subdir", "firm_summary_index")
        self.chroma_subdir = config.get("chroma_subdir", "firm_data/chroma_db")
        self.collection_name = config.get("collection_name", "firm_summary_index")
        self.force_reindex = config.get("force_reindex", False)

        self.retrieval_mode = config.get("retrieval_mode", "minilm")

        self.alpha = config.get("alpha", 0.7)

        # build derived paths
        self.output_dir = os.path.join(self.index_dir, self.output_subdir)
        self.chroma_path = os.path.join(self.index_dir, self.chroma_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

        # JSON file paths
        self.chunks_path = os.path.join(self.output_dir, "chunks.json")
        self.mapping_path = os.path.join(self.output_dir, "chunk_mapping.json")

        # Embedder + splitter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using embedding model: {self.embed_model}")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model,
            device=self.device
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        # Chroma client placeholder
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.batch_size = self.batch_size

        # In‐memory storage for Sentence Transformer
        self.all_chunks: List[str] = []
        self.all_metadatas: List[Dict[str, Any]] = []

        # Paths for BM25 artifacts
        self.bm25_model_path = os.path.join(self.output_dir, "bm25_model.pkl")
        self.bm25_tokenized_path = os.path.join(self.output_dir, "bm25_tokenized.pkl")

        # placeholders
        self.bm25_model = None
        self.bm25_tokenized = None

    def build_chunks(self, force_reindex: bool = False):
        """Load or (re)build chunks + metadata and write JSON."""
        os.makedirs(self.output_dir, exist_ok=True)

        if not force_reindex and os.path.exists(self.chunks_path) and os.path.exists(self.mapping_path):
            with open(self.chunks_path,  "r", encoding="utf-8") as f: self.all_chunks    = json.load(f)
            with open(self.mapping_path, "r", encoding="utf-8") as f: self.all_metadatas = json.load(f)
            print(f"Loaded {len(self.all_chunks)} chunks & {len(self.all_metadatas)} metadata entries.")
            return

        self.all_chunks = []
        self.all_metadatas = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Chunking summaries"):
            cid = row["hojin_id"]
            name = row["company_name"]
            kw   = row["company_keywords"]
            text = row["summary"]
            webpages = row['salient_pages']

            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                self.all_chunks.append(chunk)
                self.all_metadatas.append({
                    "company_id":       cid,
                    "company_name":     name,
                    "company_keywords": kw,
                    "webpages": webpages,
                    "chunk_index":      i
                })

        # write JSON
        with open(self.chunks_path,  "w", encoding="utf-8") as f: json.dump(self.all_chunks,    f, indent=2, ensure_ascii=False)
        with open(self.mapping_path, "w", encoding="utf-8") as f: json.dump(self.all_metadatas, f, indent=2, ensure_ascii=False)
        print(f"Built & stored {len(self.all_chunks)} chunks & {len(self.all_metadatas)} metadata entries for Firm data.")


    def build_bm25_index(self, force_rebuild: bool = False):
        # 1. If we already serialized and not forcing, just load it
        if (not force_rebuild
            and os.path.exists(self.bm25_model_path)
            and os.path.exists(self.bm25_tokenized_path)
        ):
            with open(self.bm25_model_path, "rb") as f:
                self.bm25_model = pickle.load(f)
            with open(self.bm25_tokenized_path, "rb") as f:
                self.bm25_tokenized = pickle.load(f)
            print("Loaded BM25 index from disk.")
            return

        # 2. Otherwise, (re)build from scratch
        if not self.all_chunks:
            self.build_chunks()

        # Tokenize
        tokenized = [nltk.word_tokenize(chunk.lower()) for chunk in self.all_chunks]
        self.bm25_tokenized = tokenized

        # Fit BM25
        self.bm25_model = BM25Okapi(tokenized)

        # 3. Serialize for future runs
        with open(self.bm25_model_path, "wb") as f:
            pickle.dump(self.bm25_model, f)
        with open(self.bm25_tokenized_path, "wb") as f:
            pickle.dump(self.bm25_tokenized, f)

        print(f"Built & saved BM25 model ({self.bm25_model_path}).")

    def ingest_all(self, force_reindex: bool = False):
        """(Re)create or update Chroma (dense) and BM25 (sparse) indexes."""
        # Paths for sparse index files
        bm25_model_path = os.path.join(self.output_dir, "bm25_model.pkl")
        bm25_tokenized_path = os.path.join(self.output_dir, "bm25_tokenized.pkl")

        # 1. Handle force reindex: clear dense collection and force sparse rebuild
        if force_reindex:
            # Dense: delete Chroma collection if exists
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"Deleted existing dense collection '{self.collection_name}'.")
            except Exception:
                print(f"No existing dense collection '{self.collection_name}' to delete.")
            # Sparse: force rebuild flag on BM25
            bm25_force = True
        else:
            bm25_force = False

        # 2. Check if dense exists
        dense_exists = False
        try:
            existing = self.client.get_collection(name=self.collection_name)
            dense_exists = True
        except Exception:
            dense_exists = False

        # 3. Dense exists & not forcing: skip dense rebuild
        if dense_exists and not force_reindex:
            print(f"→ Dense collection '{self.collection_name}' already exists; skipping dense ingest.")
            # ensure sparse index is present
            try:
                self.build_bm25_index(force_rebuild=False)
                print("Ensured BM25 sparse index is available.")
            except Exception as e:
                print(f"Error ensuring BM25 sparse index: {e}")
            return existing

        # 4. Otherwise, we need to (re)build both indexes
        # 4a. Ensure chunks
        self.build_chunks(force_reindex=force_reindex)

        # 4b. Build sparse first (force_rebuild if dense missing or force_reindex)
        try:
            self.build_bm25_index(force_rebuild=(bm25_force or not os.path.exists(bm25_model_path)))
            print("Built BM25 sparse index.")
        except Exception as e:
            print(f"Error building BM25 sparse index: {e}")

        # 4c. Create dense collection and ingest
        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        total = len(self.all_chunks)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            ids = [f"chunk_{i}" for i in range(start, end)]
            collection.add(
                documents=self.all_chunks[start:end],
                ids=ids,
                metadatas=self.all_metadatas[start:end]
            )
            print(f" • ingested dense batches {start}-{end}/{total}")

        print(f"Created dense collection '{self.collection_name}' with {total} chunks.")
        return collection

    def add_one(
        self,
        company_id: str,
        company_name: str,
        company_keywords: str,
        summary_text: str,
        webpages: Optional[str] = None,
    ):
        """
        Incrementally add one company's summary:
        - chunk & add to self.all_chunks / self.all_metadatas + JSON
        - ingest into Chroma
        - update BM25 tokenized corpus + model + pickle
        """
        # 1) Ensure chunks are loaded
        if not self.all_chunks:
            self.build_chunks(force_reindex=False)

        # 2) Open or create Chroma collection
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"}
            )

        # 3) Prevent duplicate company
        existing = collection.get(where={"company_id": company_id}, include=["metadatas"])
        if existing["metadatas"]:
            return {"error": f"Company {company_id} already indexed."}

        # 4) Chunk the new summary
        chunks = self.splitter.split_text(summary_text)
        offset = len(self.all_chunks)
        new_ids = [f"chunk_{offset + i}" for i in range(len(chunks))]
        new_metas = [
            {
                "company_id":       company_id,
                "company_name":     company_name,
                "company_keywords": company_keywords,
                "webpages": webpages,
                "chunk_index":      i
            }
            for i in range(len(chunks))
        ]

        # 5) Update in‑memory lists
        self.all_chunks.extend(chunks)
        self.all_metadatas.extend(new_metas)

        # 6) Persist updated chunks + metadata JSON
        with open(self.chunks_path,  "w", encoding="utf-8") as f:
            json.dump(self.all_chunks,    f, indent=2, ensure_ascii=False)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.all_metadatas, f, indent=2, ensure_ascii=False)

        # 7) Ingest into Chroma
        collection.add(
            documents=chunks,
            ids=new_ids,
            metadatas=new_metas
        )
        print(f"Added {len(chunks)} chunks for company {company_id} to Chroma; total docs = {len(self.all_chunks)}")

        # 8) --- New: Update BM25 index ---
        # 8a) Load existing tokenized corpus or initialize
        if os.path.exists(self.bm25_tokenized_path):
            with open(self.bm25_tokenized_path, "rb") as f:
                self.bm25_tokenized = pickle.load(f)
        else:
            # fresh start
            self.bm25_tokenized = [nltk.word_tokenize(c.lower()) for c in self.all_chunks[:-len(chunks)]]

        # 8b) Tokenize & append only the new chunks
        for chunk in chunks:
            self.bm25_tokenized.append(nltk.word_tokenize(chunk.lower()))

        # 8c) Rebuild BM25 model on the extended corpus
        self.bm25_model = BM25Okapi(self.bm25_tokenized)

        # 8d) Persist tokenized corpus and model
        with open(self.bm25_tokenized_path, "wb") as f:
            pickle.dump(self.bm25_tokenized, f)
        with open(self.bm25_model_path, "wb") as f:
            pickle.dump(self.bm25_model, f)

        print(f"BM25 index updated with {len(chunks)} new chunks; total docs = {len(self.bm25_tokenized)}")

        return collection


    # def retrieve_firm_contexts(
    #     self,
    #     query: str,
    #     top_k: int = 5
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Return the top-K most relevant summary chunks for `query`,
    #     each annotated with company_id, company_name, keywords, chunk_index, rank, and score.
    #     """
    #     # Make sure we have our JSON‑backed lists in memory
    #     if not self.all_chunks or not self.all_metadatas:
    #         self.build_chunks(force_reindex=False)
    #
    #     # Get (or error if missing) our Chroma collection
    #     try:
    #         collection = self.client.get_collection(name=self.collection_name)
    #     except ValueError:
    #         raise ValueError(f"Chroma collection '{self.collection_name}' for Firm not found; "
    #                          "run ingest_all() first.")
    #
    #     # Query Chroma (returns cosine *distances*)
    #     result    = collection.query(query_texts=[query], n_results=top_k)
    #     ids       = result["ids"][0]
    #     distances = result["distances"][0]
    #
    #     #  Build and return your contexts list
    #     contexts: List[Dict[str, Any]] = []
    #     for rank, (chunk_id, dist) in enumerate(zip(ids, distances), start=1):
    #         # extract original index from "chunk_{i}"
    #         idx  = int(chunk_id.split("_", 1)[1])
    #         meta = self.all_metadatas[idx]
    #
    #         contexts.append({
    #             "company_id":       meta["company_id"],
    #             "chunk":            self.all_chunks[idx],
    #             "company_name":     meta["company_name"],
    #             "company_keywords": meta["company_keywords"],
    #             "chunk_index":      meta["chunk_index"],
    #             "rank":             rank,
    #             "score":            float(1.0 - dist)
    #         })
    #
    #     return contexts

    def _retrieve_dense(self, query: str, k: int):
        """Return up to k dense hits with raw cosine scores."""
        collection = self.client.get_collection(name=self.collection_name)
        result = collection.query(query_texts=[query], n_results=k)
        ids, distances = result["ids"][0], result["distances"][0]

        hits = []
        for cid, dist in zip(ids, distances):
            try:
                idx = int(cid.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if 0 <= idx < len(self.all_chunks):
                hits.append({
                    "idx": idx,
                    "chunk": self.all_chunks[idx],
                    "meta": self.all_metadatas[idx],
                    "dense_score": 1.0 - dist
                })
        return hits

    def _retrieve_bm25(self, query: str, k: int):
        """Return up to k sparse hits with raw BM25 scores."""
        if self.bm25_model is None:
            self.build_bm25_index()

        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25_model.get_scores(tokens)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        hits = []
        for idx in top_ids:
            if 0 <= idx < len(self.all_chunks):
                hits.append({
                    "idx": idx,
                    "chunk": self.all_chunks[idx],
                    "meta": self.all_metadatas[idx],
                    "sparse_score": float(scores[idx])
                })
        return hits

    def retrieve_firm_contexts(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        candidate_k = top_k * 2

        # 0. Ensure chunks + metadata are loaded
        if not self.all_chunks or not self.all_metadatas:
            self.build_chunks(force_reindex=False)

        # 1. Get raw hits
        dense_hits = self._retrieve_dense(query, candidate_k) if self.retrieval_mode in ("minilm", "mixed") else []
        sparse_hits = self._retrieve_bm25(query, candidate_k) if self.retrieval_mode in ("bm25", "mixed") else []

        # 2. Normalize scores per set
        if dense_hits:
            scores = np.array([h["dense_score"] for h in dense_hits], dtype=float)
            lo, hi = scores.min(), scores.max()
            span = hi - lo if hi != lo else 1.0
            for h, raw in zip(dense_hits, scores):
                h["dense_norm"] = (raw - lo) / span

        if sparse_hits:
            scores = np.array([h["sparse_score"] for h in sparse_hits], dtype=float)
            lo, hi = scores.min(), scores.max()
            span = hi - lo if hi != lo else 1.0
            for h, raw in zip(sparse_hits, scores):
                h["sparse_norm"] = (raw - lo) / span

        # 3. Fuse or select based on mode
        if self.retrieval_mode == "mixed":
            # union by idx
            merged = {}
            for h in dense_hits:
                merged[h["idx"]] = {
                    **h,
                    "sparse_norm": 0.0,
                    "dense_norm": h.get("dense_norm", 0.0)
                }
            for h in sparse_hits:
                if h["idx"] in merged:
                    merged[h["idx"]]["sparse_norm"] = h.get("sparse_norm", 0.0)
                else:
                    merged[h["idx"]] = {
                        **h,
                        "dense_norm": 0.0,
                        "sparse_norm": h.get("sparse_norm", 0.0)
                    }

            combined = []
            for h in merged.values():
                fused = self.alpha * h["dense_norm"] + (1 - self.alpha) * h["sparse_norm"]
                combined.append({**h, "score": fused})

            top_hits = sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]

        elif self.retrieval_mode == "minilm":
            top_hits = sorted(dense_hits, key=lambda h: h["dense_norm"], reverse=True)[:top_k]
            for h in top_hits: h["score"] = h["dense_norm"]

        else:  # bm25 only
            top_hits = sorted(sparse_hits, key=lambda h: h["sparse_norm"], reverse=True)[:top_k]
            for h in top_hits: h["score"] = h["sparse_norm"]

        # 4. Format output
        contexts = []
        for rank, h in enumerate(top_hits, start=1):
            contexts.append({
                **h["meta"],
                "chunk": h["chunk"],
                "rank": rank,
                "score": float(h["score"])
            })
        return contexts



def main():
    # CONFIGURATION
    INDEX_DIR = r"C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\RAG_INDEX"
    firm_csv = r'C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\firms_summary_keywords_qwen.csv'

    firm_df = pd.read_csv(firm_csv)

    firm_rag = FirmSummaryRAG(
        df=firm_df,
        index_dir=INDEX_DIR,
        config=firm_config,
    )

    # Full rebuild + ingest:
    firm_rag.ingest_all(force_reindex=False)

    # # to rebuild from scratch
    # firm_rag.ingest_all(force_reindex=False)

    # Add or skip a single company summary:
    # res = firm_rag.add_one(
    #     company_id="HOJIN_1234",
    #     company_name="Acme Co.",
    #     company_keywords="robotics|ai",
    #     summary_text="Acme develops advanced robots..."
    # )
    # if isinstance(res, dict) and "error" in res:
    #     print(res["error"])

    # Query

    query = """Compositions and methods are provided that are useful for the delivery, including transdermal delivery, of biologically active agents, such as non-protein non-nucleotide therapeutics and protein-based therapeutics excluding insulin, botulinum toxins, antibody fragments, and VEGF. The compositions and methods are particularly useful for topical delivery of antifungal agents and antigenic agents suitable for immunization. Alternatively, the compositions can be prepared with components useful for targeting the delivery of the compositions as well as imaging components
    """
    results = firm_rag.retrieve_firm_contexts(query, top_k=5)
    for hit in results:
        print(f"{hit['rank']}. [{hit['score']:.3f}] {hit['company_name']} → “{hit['chunk'][:300]}…”")


if __name__ == '__main__':
    main()
