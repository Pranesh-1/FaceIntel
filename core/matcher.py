"""
Face Matcher Module
Compares a new embedding against registered embeddings using cosine similarity.
Since InsightFace returns L2-normalised embeddings, cosine similarity reduces
to a simple dot product.
"""

import numpy as np  # type: ignore
import logging

logger = logging.getLogger(__name__)


class FaceMatcher:
    def __init__(self, threshold: float = 0.45):
        """
        Args:
            threshold: Minimum cosine similarity score to call a match.
                       Range [-1, 1]; values ≥ threshold are considered the same person.
                       Recommended tuning range: 0.4 – 0.65.
        """
        self.threshold = threshold

    def match(
        self,
        query_embedding: np.ndarray,
        registered: list[tuple[str, np.ndarray]],
    ) -> tuple[str | None, float]:
        """
        Find the best match candidate for query_embedding in registered.

        Args:
            query_embedding: 1-D normalised embedding vector.
            registered: List of (face_id, embedding) tuples.

        Returns:
            (best_id, best_sim) of the closest match in the database.
            It is up to the caller (Registry) to apply thresholds/rescue logic.
        """
        if not registered:
            return None, 0.0

        best_id, best_sim = None, -1.0
        for face_id, emb in registered:
            # Cosine similarity on L2-normalised vectors == dot product
            sim = float(np.dot(query_embedding, emb))
            if sim > best_sim:
                best_sim = sim
                best_id = face_id
                
            # Early exit optimization: if we find a very strong match, stop searching
            if best_sim > 0.85:
                break

        return best_id, best_sim
