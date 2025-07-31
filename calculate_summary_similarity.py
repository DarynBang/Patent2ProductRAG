import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import random

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\sample_firms_summary_inference.csv')

print(df.head())

# 2. Load MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_summary(text: str) -> str:
    # 1) remove common indentation
    text = textwrap.dedent(text)
    # 2) collapse lines into a single space
    text = " ".join(text.splitlines())
    # 3) collapse multiple spaces, strip ends
    text = " ".join(text.split())
    # 4) lowercase if desired
    return text.lower()


# Example usage before encoding:
df['clean_gpt']  = df['summary_gpt'].apply(clean_summary)
df['clean_qwen'] = df['summary_qwen'].apply(clean_summary)


# 3. Compute embeddings for both columns
#    This returns a 2D array: shape (n_samples, embedding_dim)
embeddings_gpt  = model.encode(df['clean_gpt'].tolist(),  batch_size=32, show_progress_bar=True)
embeddings_qwen = model.encode(df['clean_qwen'].tolist(), batch_size=32, show_progress_bar=True)


# 4. Compute cosine similarity for each row
#    cos_sim[i] = cosine_similarity(embeddings_gpt[i].reshape(1,-1), embeddings_qwen[i].reshape(1,-1))[0][0]
cos_sims = cosine_similarity(embeddings_gpt, embeddings_qwen)
# cos_sims is a full matrix; diagonal elements are the rowwise similarities
rowwise_sim = np.diag(cos_sims)

# 5. Add to DataFrame
df['summary_similarity'] = rowwise_sim

# Optionally: save back out
df.to_csv(r"C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\firm_summary_with_similarity.csv", index=False)

print(df[['hojin_id', 'summary_similarity']].head())

# num_checks = 5
# n = len(sample_df)
#
# # Generate random index pairs (i, j) with i != j
# pairs = set()
# while len(pairs) < num_checks:
#     i, j = random.randrange(n), random.randrange(n)
#     if i != j:
#         pairs.add((i, j))
# pairs = list(pairs)
#
# # Compute and display their cosine similarities
# for i, j in pairs:
#     sim = cosine_similarity(
#         embeddings_gpt[i].reshape(1, -1),
#         embeddings_qwen[j].reshape(1, -1)
#     )[0, 0]
#     print(f"GPT idx={i} ↔ Qwen idx={j}  →  cosine_sim = {sim:.4f}")

