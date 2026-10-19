"""Load graph data from Parquet files."""

import json
from pathlib import Path

import pandas as pd

from skg.graph.store import GraphStore


def load_graph_from_parquet(graph_store: GraphStore, data_dir: str | Path) -> dict:
    """Load graph from Parquet files."""
    data_dir = Path(data_dir)
    stats = {
        "n_nodes_loaded": 0,
        "n_edges_loaded": 0,
        "node_types": {},
        "edge_types": {},
    }

    # Load nodes
    nodes_dir = data_dir / "nodes"
    if nodes_dir.exists():
        for node_file in nodes_dir.glob("*/*.parquet"):
            try:
                df = pd.read_parquet(node_file)
                for _, row in df.iterrows():
                    node_id = (
                        str(row.get("scenario_id", ""))
                        if "scenario_id" in row
                        else (
                            str(row.get("agent_id", ""))
                            if "agent_id" in row
                            else str(row.get("lane_id", "")) if "lane_id" in row else str(row.get("control_id", ""))
                        )
                    )
                    node_type = str(row.get("node_label", "Unknown"))
                    props = {
                        k: v
                        for k, v in row.items()
                        if k not in ["scenario_id", "agent_id", "lane_id", "control_id", "node_label"]
                    }
                    props = {k: v for k, v in props.items() if pd.notna(v)}

                    graph_store.upsert_node(node_id, node_type, props)
                    stats["n_nodes_loaded"] += 1
                    stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            except Exception as e:
                print(f"Warning: failed to load {node_file}: {e}")

    # Load edges
    edges_dir = data_dir / "edges"
    if edges_dir.exists():
        for edge_file in edges_dir.glob("*/*.parquet"):
            try:
                df = pd.read_parquet(edge_file)
                for _, row in df.iterrows():
                    src_id = str(row["src_id"])
                    src_type = str(row["src_type"])
                    rel_type = str(row["rel_type"])
                    dst_id = str(row["dst_id"])
                    dst_type = str(row["dst_type"])
                    props = {
                        k: v
                        for k, v in row.items()
                        if k not in ["src_id", "src_type", "rel_type", "dst_id", "dst_type"]
                    }
                    props = {k: v for k, v in props.items() if pd.notna(v)}

                    # Parse features JSON if present
                    if "features" in props and isinstance(props["features"], str):
                        try:
                            props["features"] = json.loads(props["features"])
                        except ValueError:
                            pass

                    graph_store.upsert_edge(src_id, src_type, rel_type, dst_id, dst_type, props)
                    stats["n_edges_loaded"] += 1
                    stats["edge_types"][rel_type] = stats["edge_types"].get(rel_type, 0) + 1
            except Exception as e:
                print(f"Warning: failed to load {edge_file}: {e}")

    return stats
