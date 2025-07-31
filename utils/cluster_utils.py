# cluster_utils.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import logging
from typing import List

MEANINGLESS_SNIPPETS = [
    "Verifying that you are not a robot...",
    "Add No.1, 3rd Area, Yangjiang Industrial Copyright2023.",
    "Some partners do not ask for your consent to process your data...",
    "Please agree to our cookie policy before continuing.",
    "© 2025 WidgetCo. All rights reserved.",
    "Our offices are located at 123 Main St, Springfield.",
    "This website uses cookies to ensure you get the best experience.",
    "Terms of service apply. See the legal page for details.",
    "Verify you are human by completing the action below.",
    "Your access to this site has been limited by the site owner."
]

MEANINGFUL_SNIPPETS = [
    "We manufacture high‑precision hydraulic pumps for automotive applications.",
    "Our core product is a modular solar battery pack rated at 5 kWh.",
    "We offer cloud‑based ERP software tailored to small businesses.",
    "The company’s flagship product is an AI‑powered image recognition API.",
    "We produce industrial‑grade conveyor belts with customizable lengths.",
    "Our service includes 24/7 remote network monitoring and alerting.",
    "We design and build custom CNC‑milled aerospace components.",
    "Our key product line is a family of noise‑cancelling over‑ear headphones.",
    "We supply advanced ceramic coatings for turbine blades.",
    "Our mobile app integrates payment processing with loyalty rewards."
]

logger = logging.getLogger(__name__)

class TextClusterFilter:
    def __init__(
        self,
        meaningless_snippets: List[str] = MEANINGLESS_SNIPPETS,
        meaningful_snippets: List[str] = MEANINGFUL_SNIPPETS,
        model_name: str = "all-mpnet-base-v2",
        n_clusters: int = 3,
    ):
        # Load model once
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

        # Embed & cluster negatives
        neg_embs = self.model.encode(meaningless_snippets, convert_to_numpy=True)
        self.neg_centroids = KMeans(n_clusters=n_clusters, random_state=42) \
                                .fit(neg_embs) \
                                .cluster_centers_
        # Embed & cluster positives
        pos_embs = self.model.encode(meaningful_snippets, convert_to_numpy=True)
        self.pos_centroids = KMeans(n_clusters=n_clusters, random_state=42) \
                                .fit(pos_embs) \
                                .cluster_centers_

    def is_meaningful(self, text: str, margin: float = 0.045) -> bool:
        if not text.strip():
            return False

        emb = self.model.encode([text], convert_to_numpy=True)[0]
        dist_neg = np.min(cosine_distances([emb], self.neg_centroids))
        dist_pos = np.min(cosine_distances([emb], self.pos_centroids))
        logger.info(f"dist_neg={dist_neg:.3f}, dist_pos={dist_pos:.3f}")

        # --- Heuristics ---
        words = text.split()
        num_words = len(words)
        unique_words = len(set(words))

        length_score = min(num_words / 100.0, 1.0)             # cap at 50 words
        unique_ratio = unique_words / num_words if num_words > 0 else 0
        richness_score = min(unique_ratio, 1.0)

        # Combine heuristic score
        boost = 0.05 * (length_score + richness_score)  # scale to small influence
        adjusted_margin = margin - boost

        logger.info(f"length_score={length_score:.3f}, richness_score={richness_score:.3f}, adjusted_margin={adjusted_margin:.3f}")

        return (dist_neg - dist_pos) > adjusted_margin
