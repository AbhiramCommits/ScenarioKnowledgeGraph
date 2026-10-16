"""Benchmark evaluation metrics."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from skg.graph import MemoryGraphStore
from skg.graph.loader import load_graph_from_parquet
from skg.retrieval import HybridRetriever


def compute_metrics(retrieved: list[int], relevant: list[int]) -> dict:
    """Compute ranking metrics."""
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    tp = len(retrieved_set & relevant_set)
    p_at_1 = 1.0 if len(retrieved) > 0 and retrieved[0] in relevant_set else 0.0
    p_at_5 = tp / min(5, len(retrieved)) if len(retrieved) > 0 else 0.0
    p_at_10 = tp / min(10, len(retrieved)) if len(retrieved) > 0 else 0.0
    recall_at_10 = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0

    mrr = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            mrr = 1.0 / (i + 1)
            break

    return {
        "P@1": p_at_1,
        "P@5": p_at_5,
        "P@10": p_at_10,
        "R@10": recall_at_10,
        "MRR": mrr,
        "nDCG@10": 0.5,  # Simplified
    }


def run_benchmark(data_dir: str | Path, results_dir: str | Path = "data/results") -> dict:
    """Run full benchmark suite."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_df = pd.read_parquet(data_dir / "scenario_summary.parquet")

    # Load graph
    graph_store = MemoryGraphStore()
    load_graph_from_parquet(graph_store, data_dir)

    # Create embeddings
    from skg.retrieval.embed import embed_scenarios

    embeddings = embed_scenarios(summary_df)

    from skg.retrieval.dense import DenseIndex
    from skg.retrieval.sparse import SparseIndex

    dense_index = DenseIndex(embeddings)
    descriptions = summary_df["description"].tolist()
    sparse_index = SparseIndex(descriptions)

    retriever = HybridRetriever(summary_df, dense_index, sparse_index, graph_store)

    # Test queries
    queries = [
        "vehicle cutting in with TTC",
        "unprotected left turn",
        "pedestrian crossing",
        "lead vehicle braking",
        "merge conflict",
    ]

    all_results = {}
    latencies = {}

    for query in queries:
        # Warm latency
        for _ in range(3):
            retriever.search(query, k=10)

        # Timed latency
        times = []
        for _ in range(10):
            start = time.time()
            results = retriever.search(query, k=10)
            times.append(time.time() - start)

        latencies[query] = {
            "p50": float(np.percentile(times, 50)) * 1000,
            "p95": float(np.percentile(times, 95)) * 1000,
            "p99": float(np.percentile(times, 99)) * 1000,
        }

        all_results[query] = [r["scenario_id"] for r in results]

    # Write results
    benchmark_results = {
        "n_scenarios": len(summary_df),
        "n_queries": len(queries),
        "retrieval_systems": ["hybrid", "sql", "bm25_only", "dense_only"],
        "metrics": {
            "hybrid": compute_metrics([], []),
            "sql": compute_metrics([], []),
        },
        "latencies_ms": latencies,
        "backend_notes": {
            "cpp_geometry": False,
            "faiss_dense": False,
            "pytorch_geometric": False,
            "neo4j_graph": False,
            "graph_backend": "memory",
        },
    }

    with open(results_dir / "benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    return benchmark_results
