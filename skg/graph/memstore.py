"""In-memory graph store using NetworkX."""

from typing import Any

import networkx as nx

from skg.graph.store import GraphStore


class MemoryGraphStore(GraphStore):
    """Pure-Python in-memory graph store."""

    def __init__(self) -> None:
        """Initialize."""
        self.graph = nx.MultiDiGraph()
        self.nodes_by_type: dict[str, set[str]] = {}

    def upsert_node(self, node_id: str, node_type: str, properties: dict[str, Any]) -> None:
        """Upsert a node."""
        # Avoid conflict with 'type' parameter
        props = {k: v for k, v in properties.items() if k != "type"}
        props["node_type"] = node_type
        self.graph.add_node(node_id, type=node_type, **props)
        if node_type not in self.nodes_by_type:
            self.nodes_by_type[node_type] = set()
        self.nodes_by_type[node_type].add(node_id)

    def upsert_edge(
        self,
        src_id: str,
        src_type: str,
        rel_type: str,
        dst_id: str,
        dst_type: str,
        properties: dict[str, Any],
    ) -> None:
        """Upsert an edge."""
        # Ensure nodes exist
        if src_id not in self.graph:
            self.upsert_node(src_id, src_type, {})
        if dst_id not in self.graph:
            self.upsert_node(dst_id, dst_type, {})

        self.graph.add_edge(src_id, dst_id, rel_type=rel_type, **properties)

    def motif_search(self, pattern: str, params: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Simple motif search over in-memory graph."""
        results = []

        # Extract motif type from pattern string
        target_rel = None
        if "NEAR_MISS" in pattern:
            target_rel = "NEAR_MISS"
        elif "TRAVELS_ON_LANE" in pattern:
            target_rel = "TRAVELS_ON_LANE"
        elif "CUTS_IN" in pattern:
            target_rel = "CUTS_IN"
        elif "APPROACHES" in pattern:
            target_rel = "APPROACHES"

        if target_rel:
            # Find edges matching the target relation type
            for src, dst, key, data in self.graph.out_edges(keys=True, data=True):
                if data.get("rel_type") == target_rel:
                    results.append({"src": src, "dst": dst, "type": target_rel})
                    if len(results) >= limit:
                        break

        return results

    def k_hop_neighbors(self, node_id: str, k: int = 2) -> list[tuple[str, str, int]]:
        """Get k-hop neighbors."""
        results = []

        if node_id not in self.graph:
            return results

        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if dist > 0:
                results.append((current, "neighbor", dist))

            if dist < k:
                for neighbor in self.graph.successors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

        return results

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Get node properties."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_edge_properties(self, src_id: str, rel_type: str, dst_id: str) -> dict[str, Any] | None:
        """Get edge properties."""
        if self.graph.has_edge(src_id, dst_id):
            edges = self.graph[src_id][dst_id]
            for data in edges.values():
                if data.get("rel_type") == rel_type:
                    return dict(data)
        return None

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "n_nodes_by_type": {k: len(v) for k, v in self.nodes_by_type.items()},
            "backend": "memory",
        }

    def clear(self) -> None:
        """Clear all data."""
        self.graph.clear()
        self.nodes_by_type.clear()
