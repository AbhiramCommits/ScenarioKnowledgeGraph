"""Graph store abstract interface."""

from abc import ABC, abstractmethod
from typing import Any


class GraphStore(ABC):
    """Abstract graph store interface."""

    @abstractmethod
    def upsert_node(self, node_id: str, node_type: str, properties: dict[str, Any]) -> None:
        """Upsert a node."""

    @abstractmethod
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

    @abstractmethod
    def motif_search(
        self, pattern: str, params: dict[str, Any] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Search for motif patterns. Pattern is a Cypher-like query."""

    @abstractmethod
    def k_hop_neighbors(self, node_id: str, k: int = 2) -> list[tuple[str, str, str]]:
        """Get k-hop neighbors. Returns list of (node_id, rel_type, distance)."""

    @abstractmethod
    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Get node properties."""

    @abstractmethod
    def get_edge_properties(self, src_id: str, rel_type: str, dst_id: str) -> dict[str, Any] | None:
        """Get edge properties."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return store statistics."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
