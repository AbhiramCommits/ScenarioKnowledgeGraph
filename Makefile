.PHONY: install test smoke bench all clean help

help:
	@echo "ScenarioKnowledgeGraph targets:"
	@echo "  install    Install dependencies"
	@echo "  test       Run test suite"
	@echo "  smoke      Run smoke test (200 scenarios)"
	@echo "  bench      Run full benchmark (10k scenarios)"
	@echo "  all        Generate, extract, index, benchmark"
	@echo "  clean      Remove generated data"

install:
	pip install -q -e .

test:
	pytest tests/ -v

smoke:
	python -m skg.logs.generator --n 200 --out data/raw_smoke --seed 7
	python -m skg.extract.spark_job --in data/raw_smoke --out data/graph_smoke
	python -c "from skg.graph import MemoryGraphStore; from skg.graph.loader import load_graph_from_parquet; s = MemoryGraphStore(); load_graph_from_parquet(s, 'data/graph_smoke'); print(f'Smoke test: {s.stats()}')"

bench:
	python scripts/run_full_benchmark.py

all: install test smoke bench
	@echo "All targets completed"

clean:
	rm -rf data/{raw,raw_smoke,graph,graph_smoke,index,models,results}/*.{parquet,pkl,npy,json}
	rm -rf build dist *.egg-info
