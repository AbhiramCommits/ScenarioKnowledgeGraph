"""Sparse vector search using BM25."""

import pickle
from pathlib import Path


class SparseIndex:
    """Sparse BM25 search index."""

    def __init__(self, documents: list[str]) -> None:
        """Initialize from document strings."""
        self.documents = documents
        self.n_docs = len(documents)

        try:
            from rank_bm25 import BM25Okapi

            # Tokenize
            tokenized = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized)
            self.backend = "bm25"
        except ImportError:
            self.bm25 = None
            self.backend = "fallback"

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Search for top-k documents."""
        if self.bm25 is not None:
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
            return [(int(idx), float(score)) for idx, score in top_k]
        else:
            # Fallback: substring matching
            query_terms = query.lower().split()
            scores = []
            for doc in self.documents:
                score = sum(1 for term in query_terms if term in doc.lower())
                scores.append(score)
            top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
            return [(int(idx), float(score)) for idx, score in top_k]

    def save(self, path: str | Path) -> None:
        """Save index."""
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str | Path) -> "SparseIndex":
        """Load index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(data["documents"])
        idx.bm25 = data.get("bm25")
        return idx
