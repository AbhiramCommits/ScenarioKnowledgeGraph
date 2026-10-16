"""Tests for graph store."""

import tempfile
from pathlib import Path

import pytest

from skg.extract.spark_job import run_extraction
from skg.graph.loader import load_graph_from_parquet
from skg.graph.memstore import MemoryGraphStore
from skg.logs.generator import generate_corpus


@pytest.fixture
def memory_store() -> MemoryGraphStore:
    """Create empty memory store."""
    return MemoryGraphStore()


@pytest.fixture
def populated_store(memory_store: MemoryGraphStore) -> MemoryGraphStore:
    """Create populated store with sample data."""
    memory_store.upsert_node("scenario_1", "Scenario", {"label": "CUT_IN"})
    memory_store.upsert_node("agent_1", "Agent", {"type": "VEHICLE"})
    memory_store.upsert_node("agent_2", "Agent", {"type": "VEHICLE"})
    memory_store.upsert_node("lane_1", "Lane", {"speed_limit": 25})

    memory_store.upsert_edge("agent_1", "Agent", "CUTS_IN", "agent_2", "Agent", {"min_gap": 2.5})
    memory_store.upsert_edge("agent_1", "Agent", "TRAVELS_ON_LANE", "lane_1", "Lane", {})

    return memory_store


def test_upsert_node(memory_store: MemoryGraphStore) -> None:
    """Test node upsert."""
    memory_store.upsert_node("n1", "Test", {"attr": "val"})
    props = memory_store.get_node_properties("n1")
    assert props is not None
    assert props["type"] == "Test"
    assert props["attr"] == "val"


def test_upsert_edge(memory_store: MemoryGraphStore) -> None:
    """Test edge upsert."""
    memory_store.upsert_node("a", "Type1", {})
    memory_store.upsert_node("b", "Type2", {})
    memory_store.upsert_edge("a", "Type1", "REL", "b", "Type2", {"weight": 1.5})

    props = memory_store.get_edge_properties("a", "REL", "b")
    assert props is not None
    assert props["weight"] == 1.5


def test_k_hop_neighbors(populated_store: MemoryGraphStore) -> None:
    """Test k-hop neighbor search."""
    neighbors = populated_store.k_hop_neighbors("agent_1", k=2)
    neighbor_ids = [n[0] for n in neighbors]
    assert "agent_2" in neighbor_ids or "lane_1" in neighbor_ids


def test_motif_search(populated_store: MemoryGraphStore) -> None:
    """Test motif search."""
    results = populated_store.motif_search("CUTS_IN")
    assert len(results) > 0
    assert any(r.get("type") == "CUTS_IN" for r in results)


def test_stats(populated_store: MemoryGraphStore) -> None:
    """Test statistics."""
    stats = populated_store.stats()
    assert stats["n_nodes"] == 4
    assert stats["n_edges"] == 2
    assert "Scenario" in stats["n_nodes_by_type"]


def test_clear(populated_store: MemoryGraphStore) -> None:
    """Test clearing store."""
    populated_store.clear()
    stats = populated_store.stats()
    assert stats["n_nodes"] == 0
    assert stats["n_edges"] == 0


def test_load_from_extraction() -> None:
    """Test loading graph from extracted data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate corpus
        generate_corpus(n_scenarios=2, out_dir=tmpdir, seed=42, n_agents=3, duration_steps=20)

        # Extract
        extract_dir = Path(tmpdir) / "extracted"
        run_extraction(tmpdir, str(extract_dir))

        # Load into store
        store = MemoryGraphStore()
        stats = load_graph_from_parquet(store, extract_dir)

        assert stats["n_nodes_loaded"] > 0
        assert stats["n_edges_loaded"] >= 0
        store_stats = store.stats()
        assert store_stats["n_nodes"] > 0
