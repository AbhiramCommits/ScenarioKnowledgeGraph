"""Hybrid retrieval engine fusing multiple search legs."""

from pathlib import Path

import numpy as np
import pandas as pd

from skg.graph import MemoryGraphStore
from skg.retrieval.dense import DenseIndex
from skg.retrieval.fusion import reciprocal_rank_fusion
from skg.retrieval.sparse import SparseIndex


class HybridRetriever:
    """Hybrid retrieval engine."""

    def __init__(
        self,
        summary_df: pd.DataFrame,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        graph_store: MemoryGraphStore,
    ) -> None:
        """Initialize retriever."""
        self.summary_df = summary_df.reset_index(drop=True)
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.graph_store = graph_store
        self.scenario_ids = list(self.summary_df["scenario_id"])

    def search(
        self,
        query: str,
        k: int = 10,
        legs: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[dict]:
        """Search for scenarios."""
        if legs is None:
            legs = ["dense", "sparse", "graph"]

        rankings = {}

        # Dense leg
        if "dense" in legs:
            query_embedding = np.array([len(query.split()), 50, 5.0, 15.0], dtype=np.float32)
            mean = np.array([7.0, 50.0, 5.0, 15.0])
            std = np.array([2.0, 10.0, 2.0, 3.0]) + 1e-6
            query_embedding = (query_embedding - mean) / std
            dense_results = self.dense_index.search(query_embedding, k=50)
            rankings["dense"] = dense_results

        # Sparse leg
        if "sparse" in legs:
            sparse_results = self.sparse_index.search(query, k=50)
            rankings["sparse"] = sparse_results

        # Graph leg
        if "graph" in legs:
            graph_results = []
            for motif in ["near_miss", "cut_in_then_brake"]:
                motif_results = self.graph_store.motif_search(motif, limit=25)
                for r in motif_results:
                    graph_results.append((hash(r.get("src", "")) % len(self.scenario_ids), 1.0))
            if not graph_results:
                graph_results = [(i, 0.5) for i in range(min(10, len(self.scenario_ids)))]
            rankings["graph"] = graph_results[:50]

        # Fuse rankings
        fused = reciprocal_rank_fusion(rankings, weights=weights, top_k=k)

        # Build results
        results = []
        for scenario_idx, score in fused:
            if scenario_idx < len(self.scenario_ids):
                row = self.summary_df.iloc[scenario_idx]
                results.append(
                    {
                        "scenario_id": str(row.get("scenario_id", "")),
                        "behavior_label": str(row.get("behavior_label", "")),
                        "score": float(score),
                        "rank_contributions": {
                            leg: idx
                            for leg, res in rankings.items()
                            for idx, (sid, _) in enumerate(res)
                            if sid == scenario_idx
                        },
                    }
                )

        return results[:k]

    def save(self, path: str | Path) -> None:
        """Save retriever state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dense_index.save(path / "dense.npy")
        self.sparse_index.save(path / "sparse.pkl")
        self.summary_df.to_parquet(path / "summary.parquet")

    @classmethod
    def load(cls, path: str | Path, graph_store: MemoryGraphStore) -> "HybridRetriever":
        """Load retriever from disk."""
        path = Path(path)
        dense_index = DenseIndex.load(path / "dense.npy")
        sparse_index = SparseIndex.load(path / "sparse.pkl")
        summary_df = pd.read_parquet(path / "summary.parquet")
        return cls(summary_df, dense_index, sparse_index, graph_store)
