# CLEVER — Cluster-Level Eviction for Vector Embedding Retrieval

Benchmarking and optimization framework for semantic caching in LLM applications.

**Course:** CSE 584 — Advanced Database Systems, University of Michigan
**Team:** Yash Kulkarni, Shubham Harkare, Arvind Suresh

## Overview

CLEVER evaluates ANN index structures (HNSW, IVF, LSH, Flat) for semantic caching under realistic LLM workloads. It implements cost-based query routing, proposes semantic-aware eviction, and analyzes scalability from 10K to 1M cached entries.

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter project
git clone <repo-url>
cd CLEVER

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
bash scripts/00_setup_environment.sh
```

### 2. Download & Preprocess Data

```bash
# Login to HuggingFace (required for LMSYS-Chat-1M access)
huggingface-cli login

# Download and preprocess (dev mode — first 10K rows)
python scripts/01_download_dataset.py --output data/ --max-rows 10000

# Full dataset (run on Great Lakes or with patience)
python scripts/01_download_dataset.py --output data/
```

### 3. Generate Embeddings

```bash
# Local (CPU, small subsets for development)
python scripts/02_generate_embeddings.py \
    --input data/processed_queries.parquet \
    --output results/embeddings/ \
    --device cpu --batch-size 64 \
    --sizes 10k,50k

# Great Lakes (GPU, all subsets)
sbatch slurm/embed.sbatch
```

## Project Structure

```
CLEVER/
├── src/                     # Core library
│   ├── data/                # Data loading, preprocessing, sampling
│   ├── embeddings/          # Sentence-transformers encoder
│   ├── indexes/             # FAISS index wrappers (Flat, HNSW, IVF, LSH)
│   ├── cache/               # Semantic cache + eviction policies
│   ├── benchmark/           # Metrics, workload generation, profiling
│   └── evaluation/          # LLM-as-judge, analysis
├── scripts/                 # CLI entry points (one per phase)
├── configs/                 # YAML experiment configurations
├── slurm/                   # Great Lakes HPC job scripts
├── tests/                   # pytest test suite
├── results/                 # Outputs (embeddings, benchmarks, figures)
└── data/                    # Processed datasets (parquet)
```

## Experiment Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `00_setup_environment.sh` | Verify environment |
| 1 | `01_download_dataset.py` + `02_generate_embeddings.py` | Data pipeline |
| 2 | `03_run_index_benchmark.py` | Index comparison (HNSW, IVF, LSH, Flat) |
| 3 | `04_run_cost_model.py` | Cost-based query routing |
| 4 | `05_run_eviction.py` | Semantic-aware eviction |
| 5 | `06_run_scalability.py` | Scalability analysis (10K–1M) |
| 6 | `07_llm_judge.py` | Semantic validity check |
| 7 | `08_generate_figures.py` | Plots and tables for paper |

## Running Tests

```bash
pytest tests/ -v
```

## Hardware Requirements

- **Local (MacBook):** Development, tests, ≤50K vectors. CPU-only FAISS.
- **Great Lakes HPC:** Full experiments (100K–1M). GPU for embeddings, CPU for benchmarks.
