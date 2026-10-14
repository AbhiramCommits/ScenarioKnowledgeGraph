"""Reciprocal Rank Fusion for multi-leg retrieval."""

from typing import Optional


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, float]]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Fuse rankings using RRF."""
    if weights is None:
        weights = {name: 1.0 for name in rankings.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Compute RRF scores
    scores: dict[int, float] = {}

    for leg_name, ranked_results in rankings.items():
        weight = weights[leg_name]
        for rank, (doc_id, orig_score) in enumerate(ranked_results):
            rrf_score = weight * (1.0 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

    # Sort and return top-k
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results
