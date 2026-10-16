"""Tests for retrieval engine."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from skg.baselines.sql_baseline import SQLBaseline
from skg.graph import MemoryGraphStore
from skg.logs.generator import generate_corpus
from skg.retrieval.dense import DenseIndex
from skg.retrieval.embed import embed_scenarios
from skg.retrieval.engine import HybridRetriever
from skg.retrieval.fusion import reciprocal_rank_fusion
from skg.retrieval.sparse import SparseIndex


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample scenario dataframe."""
    return pd.DataFrame(
        {
            "scenario_id": [f"scen_{i}" for i in range(100)],
            "behavior_label": ["CUT_IN", "BRAKE", "YIELD"] * 33 + ["CUT_IN"],
            "description": [f"Scenario {i}" for i in range(100)],
            "n_agents": [5] * 100,
            "duration_steps": [50] * 100,
            "min_ttc": [3.0 + i * 0.01 for i in range(100)],
            "max_speed": [15.0 + i * 0.1 for i in range(100)],
        }
    )


def test_embed_scenarios(sample_df: pd.DataFrame) -> None:
    """Test scenario embedding."""
    embeddings = embed_scenarios(sample_df)
    assert embeddings.shape == (100, 4)
    assert embeddings.dtype == "float32"


def test_dense_index(sample_df: pd.DataFrame) -> None:
    """Test dense vector search."""
    embeddings = embed_scenarios(sample_df)
    index = DenseIndex(embeddings)

    query = embeddings[0]
    results = index.search(query, k=10)
    assert len(results) == 10
    assert results[0][0] == 0  # Top result should be the query itself


def test_sparse_index(sample_df: pd.DataFrame) -> None:
    """Test BM25 sparse search."""
    index = SparseIndex(sample_df["description"].tolist())
    results = index.search("Scenario 5", k=10)
    assert len(results) > 0


def test_rrf_fusion() -> None:
    """Test RRF ranking fusion."""
    rankings = {
        "leg1": [(1, 10.0), (2, 9.0), (3, 8.0)],
        "leg2": [(3, 10.0), (1, 5.0), (2, 4.0)],
    }
    fused = reciprocal_rank_fusion(rankings, top_k=3)
    assert len(fused) == 3
    assert isinstance(fused[0][0], int)


def test_hybrid_retriever(sample_df: pd.DataFrame) -> None:
    """Test hybrid retriever."""
    embeddings = embed_scenarios(sample_df)
    dense = DenseIndex(embeddings)
    sparse = SparseIndex(sample_df["description"].tolist())
    graph_store = MemoryGraphStore()

    retriever = HybridRetriever(sample_df, dense, sparse, graph_store)
    results = retriever.search("cut in", k=10)

    assert len(results) <= 10
    for result in results:
        assert "scenario_id" in result
        assert "score" in result


def test_sql_baseline(sample_df: pd.DataFrame) -> None:
    """Test SQL baseline."""
    baseline = SQLBaseline(sample_df)
    results = baseline.search("cut", k=5)
    assert len(results) <= 5
    baseline.close()


def test_end_to_end_retrieval() -> None:
    """End-to-end retrieval test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate
        generate_corpus(n_scenarios=20, out_dir=tmpdir, seed=42, n_agents=3, duration_steps=20)

        # Load summary from generator output
        summary_df = pd.read_parquet(Path(tmpdir) / "scenario_summary.parquet")

        # Build retriever
        embeddings = embed_scenarios(summary_df)
        dense = DenseIndex(embeddings)
        sparse = SparseIndex(summary_df["description"].tolist())
        graph_store = MemoryGraphStore()

        retriever = HybridRetriever(summary_df, dense, sparse, graph_store)
        results = retriever.search("vehicle", k=10)

        assert len(results) > 0
        assert all("scenario_id" in r for r in results)
