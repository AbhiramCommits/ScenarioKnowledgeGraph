"""Dense vector search using FAISS."""

from pathlib import Path

import numpy as np


class DenseIndex:
    """Dense vector search index."""

    def __init__(self, embeddings: np.ndarray) -> None:
        """Initialize from embeddings."""
        self.embeddings = embeddings
        self.n_scenarios = len(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for top-k similar scenarios."""
        # L2 distance
        distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return [(int(idx), float(distances[idx])) for idx in top_k_indices]

    def save(self, path: str | Path) -> None:
        """Save index."""
        np.save(path, self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        """Load index."""
        embeddings = np.load(path)
        return cls(embeddings)
