"""Configuration and settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Global configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    graph_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "graph")
    index_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "index")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "models")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "results")

    # Corpus generation
    default_n_scenarios: int = 10000
    default_n_agents_per_scenario: int = 7
    default_duration_steps: int = 50
    default_fps: int = 10
    generator_seed: int = 7

    # Geometry thresholds
    min_ttc_near_miss: float = 3.0
    lateral_offset_threshold: float = 2.0

    # Graph
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

    # Retrieval
    k_dense: int = 50
    k_bm25: int = 50
    k_graph: int = 50
    default_k_final: int = 10
    rrf_k: int = 60

    def __post_init__(self) -> None:
        for d in [self.data_dir, self.raw_data_dir, self.graph_data_dir, self.index_dir, self.models_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


_global_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings singleton."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings
