# ScenarioKnowledgeGraph

Bi-temporal scenario knowledge graph for autonomous driving: converts raw multi-agent logs into structured scenarios with semantic relations, indexes them with hybrid retrieval (graph + dense vectors + BM25), and provides a benchmark against SQL tabular baselines.

## What It Does

1. **Synthetic Data Generation**: Creates 10,000 realistic multi-agent driving scenarios (Argoverse 2 style) with 8 labeled behavior types.
2. **Semantic Extraction**: Spark job extracts typed entities (Scenario, Agent, Lane, TrafficControl) and timestamped relations (TRAVELS_ON_LANE, CUTS_IN, NEAR_MISS, etc.) from raw logs.
3. **Bi-Temporal Graph**: Stores world-time and system-time versioning for scenarios. Supports as-of queries and retraction semantics.
4. **Hybrid Retrieval**: Fuses graph traversal (motif search), dense scenario embeddings (L2 distance), and sparse BM25 keyword search via reciprocal rank fusion.
5. **Baselines**: SQL tabular indexing and retrieval for quality/latency comparison.
6. **Benchmark**: Retrieval metrics (P@k, R@k, MRR, nDCG) and latency profiles (p50/p95/p99) on 10k scenarios.

## Architecture

```
Raw Logs (Parquet)
       |
       v
   Generator
       |
       v
Semantic Extraction (Spark)
  - AgentTrack parsing
  - Kinematic feature computation (C++/NumPy)
  - Relation labeling (CUTS_IN, NEAR_MISS, etc.)
       |
       v
Typed Entity Nodes + Timestamped Edges (Parquet)
       |
       v
Graph Loading (MemoryGraphStore | Neo4j)
       |
       v
Retrieval Index Building
  - Scenario Embeddings (kinematics → 4-dim L2-normalized)
  - Dense: DenseIndex (L2 distance search)
  - Sparse: BM25 + fallback substring matching
  - Graph: Motif search (k-hop, pattern matching)
       |
       v
HybridRetriever (RRF fusion)
       |
       v
Metrics + Latency Benchmark
```

## Installation

**Requirements**: Python 3.11, Linux/macOS, Docker (optional for Neo4j)

```bash
git clone <repo> && cd ScenarioKnowledgeGraph
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**With Neo4j** (optional):
```bash
docker compose up neo4j
export NEO4J_URI=bolt://localhost:7687
```

## Usage

### Generate Synthetic Corpus
```bash
python -m skg.logs.generator --n 10000 --out data/raw --seed 7
# Output: data/raw/scenarios/*.parquet (2.75M track points, 10k scenarios)
```

### Extract Semantics
```bash
python -m skg.extract.spark_job --in data/raw --out data/graph
# Output: data/graph/{nodes,edges}/*.parquet, stats.json
# Spark extraction: 116k rows/sec on default config
```

### Load Graph
```bash
from skg.graph import MemoryGraphStore
from skg.graph.loader import load_graph_from_parquet
store = MemoryGraphStore()
load_graph_from_parquet(store, "data/graph")
print(store.stats())  # 105k nodes, 100k edges
```

### Build Retrieval Index
```bash
import pandas as pd
from skg.retrieval import HybridRetriever
from skg.retrieval.embed import embed_scenarios
from skg.retrieval.dense import DenseIndex
from skg.retrieval.sparse import SparseIndex

summary = pd.read_parquet("data/raw/scenario_summary.parquet")
embeddings = embed_scenarios(summary)
dense = DenseIndex(embeddings)
sparse = SparseIndex(summary["description"].tolist())

retriever = HybridRetriever(summary, dense, sparse, store)
results = retriever.search("cut-in with TTC", k=10)
# results: [{"scenario_id": "...", "behavior_label": "...", "score": ...}]
```

### Run Benchmark
```bash
python scripts/run_full_benchmark.py
# Outputs: data/results/benchmark.json
```

### Run Tests
```bash
make test     # All tests (geometry, extraction, graph, retrieval)
make smoke    # 200-scenario smoke test
make bench    # Full 10k-scenario benchmark
```

## Results

**Corpus**: 10,000 scenarios, 2.75M track points, 105k entity nodes, 100k timestamped edges

**Extraction** (Spark, local[8]):
- Wall-clock: 23.65s
- Throughput: 116k rows/sec
- Nodes: 10k Scenarios + 55k Agents + 30k Lanes + 10k Controls
- Edges: 100k (TRAVELS_ON_LANE, APPROACHES, NEAR_MISS by kinematic thresholds)

**Retrieval Latency** (10 queries × 10 runs each, warm):
| Query | p50 ms | p95 ms | p99 ms |
|-------|--------|--------|--------|
| cut-in with TTC | 4.4 | 17.3 | 20.1 |
| unprotected left turn | 4.0 | 10.1 | 12.5 |
| pedestrian crossing | 5.5 | 12.9 | 15.6 |
| lead vehicle braking | 4.5 | 16.9 | 19.9 |
| merge conflict | 3.3 | 10.6 | 11.0 |

**Backends Used**:
- Geometry: NumPy fallback (C++ pybind11 extension not built; would need `cmake`)
- Dense search: Pure NumPy L2 distance (FAISS not installed; CPU-only)
- Graph store: NetworkX in-memory (Neo4j optional; doccker-compose config provided)
- Sparse search: rank_bm25 with substring fallback

**Hybrid Retrieval**: RRF fusion of dense + sparse + graph legs achieves ~4-5ms p50 latency on 10k corpus.

## Key Design Decisions

1. **Bi-Temporal Edges**: `valid_from/valid_to` (world time) + `ingested_at/retracted_at` (system time) enable scenario recovery at past ingestion points and safe schema migration.

2. **Fallback Implementations**: C++/FAISS/PyTorch unavailable → automatic NumPy/NetworkX/substring equivalents. Backend state reported in benchmark output.

3. **RRF Fusion**: Simple, interpretable weighting; supports leg subsets for ablation (dense-only, sparse-only, graph-only, any combination).

4. **Kinematic Features**: Computed once during extraction; indexed for fast retrieval. TTC, min gap, closing rate, lateral offset from centerline.

5. **Synthetic Labels**: Behavior drawn from 8-type taxonomy; descriptions templated with real kinematics (speeds, gaps, TTC) so BM25 queries are meaningful.

## Limitations

- **Synthetic Data**: 10k scenarios ≠ real Argoverse 2 / nuScenes distribution. Trajectories generated from simple kinematic patterns, not learning-based models.
- **Scale**: Spark extraction on single machine. Cluster deployment would be straightforward (same code, different Spark config).
- **GNN Baseline**: Placeholder only (no PyTorch Geometric link-prediction training in this build).
- **Learned Fusion**: RRF weights fixed; could be learned from ranking losses.

## Roadmap

1. **Real Data**: Integrate Argoverse 2 / nuScenes / Lyft Level 5 native loaders.
2. **Learned Fusion**: LambdaMART or pointwise NN fusion.
3. **Graph-RAG Agent**: Use hybrid retriever as memory for LLM-guided scenario analysis.
4. **Cluster Scaling**: Spark on Dataproc / EMR, Neo4j on managed Aura, FAISS on GPU.
5. **Web UI**: Flask / React dashboard for scenario search & graph exploration.

## Citation

If you use this codebase, cite:
```
@software{skg2026,
  title={ScenarioKnowledgeGraph: Bi-Temporal Driving Scenario Knowledge Graph},
  author={Anonymous},
  year={2026},
  url={https://github.com/...}
}
```

## License

MIT
