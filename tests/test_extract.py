"""Tests for semantic extraction pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from skg.extract.spark_job import run_extraction
from skg.logs.generator import generate_corpus


@pytest.fixture
def small_corpus() -> str:
    """Generate small test corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_corpus(n_scenarios=5, out_dir=tmpdir, seed=42, n_agents=3, duration_steps=20)
        yield tmpdir


def test_extraction_runs(small_corpus: str) -> None:
    """Test that extraction completes without error."""
    with tempfile.TemporaryDirectory() as out_dir:
        stats = run_extraction(small_corpus, out_dir)

        assert stats["n_scenarios"] == 5
        assert stats["n_agents"] > 0
        assert stats["n_edges"] > 0
        assert stats["wall_clock_seconds"] > 0

        # Verify output structure
        out_path = Path(out_dir)
        assert (out_path / "nodes").exists()
        assert (out_path / "edges").exists()
        assert (out_path / "stats.json").exists()


def test_node_invariants(small_corpus: str) -> None:
    """Test node extraction invariants."""
    with tempfile.TemporaryDirectory() as out_dir:
        run_extraction(small_corpus, out_dir)

        out_path = Path(out_dir)

        # Read nodes
        scenarios_path = list((out_path / "nodes" / "scenarios").glob("*.parquet"))[0]
        agents_path = list((out_path / "nodes" / "agents").glob("*.parquet"))[0]

        scenarios = pd.read_parquet(scenarios_path)
        agents = pd.read_parquet(agents_path)

        # Scenarios
        assert "scenario_id" in scenarios.columns
        assert "behavior_label" in scenarios.columns
        assert len(scenarios) == 5

        # Agents
        assert "agent_id" in agents.columns
        assert "agent_type" in agents.columns
        assert len(agents) > 0


def test_edge_invariants(small_corpus: str) -> None:
    """Test edge extraction invariants."""
    with tempfile.TemporaryDirectory() as out_dir:
        run_extraction(small_corpus, out_dir)

        out_path = Path(out_dir)
        edges_path = list((out_path / "edges" / "all_relations").glob("*.parquet"))[0]

        edges = pd.read_parquet(edges_path)

        # Check required columns
        required_cols = [
            "src_id",
            "src_type",
            "rel_type",
            "dst_id",
            "dst_type",
            "scenario_id",
            "valid_from_ms",
            "valid_to_ms",
        ]
        for col in required_cols:
            assert col in edges.columns, f"Missing column: {col}"

        # No self-edges
        assert not (edges["src_id"] == edges["dst_id"]).any(), "Self-edges found"

        # valid_from < valid_to
        assert (edges["valid_from_ms"] <= edges["valid_to_ms"]).all(), "Invalid time ranges"

        # All edges have endpoints
        assert len(edges) > 0
