# CLAUDE.md — Semantic Cache Index Benchmarking Project

## Project Overview

**Title:** Index Structures and Query Processing for Semantic Cache Systems in LLM Applications
**Course:** CSE 584 — Advanced Database Systems, University of Michigan
**Team:** Yash Kulkarni, Shubham Harkare, Arvind Suresh

We are building a comprehensive benchmarking and optimization framework for semantic caching in LLM applications. The project evaluates ANN index structures (HNSW, IVF, LSH, Flat) under realistic LLM workloads, implements cost-based query routing, proposes semantic-aware eviction, and analyzes scalability from 10K to 1M cached entries.

---

## Hardware & Environment

### Local Machine (MacBook Air M4, 16GB RAM)
- Use for: development, debugging, small-scale tests (≤50K vectors), visualization
- Python 3.11+, no GPU — use CPU-only FAISS (`faiss-cpu`)
- Memory ceiling: keep peak RAM usage under 12GB to avoid swap

### Great Lakes HPC (University of Michigan)
- Use for: full dataset embedding generation, all benchmark experiments at scale (100K–1M), final reproducible runs
- GPU nodes available (A100/V100) for embedding generation
- SLURM scheduler — all jobs must be submitted via `sbatch`
- Module system: `module load python/3.11 cuda/12.1`

### Decision Rule
- **Local:** Any task with ≤50K vectors, code development, unit tests, plotting
- **Great Lakes:** Embedding 1M conversations, any experiment with >50K vectors, final benchmark runs for the paper

---

## Project Structure

```
semantic-cache-benchmark/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview, setup, reproduction steps
├── pyproject.toml               # Project dependencies (use uv or pip)
├── .env.example                 # Template for API keys (OpenAI for LLM-as-judge)
│
├── configs/                     # All experiment configs (YAML)
│   ├── index_benchmark.yaml     # Index types, parameters, dataset sizes
│   ├── cost_model.yaml          # Cost model parameters
│   ├── eviction.yaml            # Eviction policy configs
│   └── scalability.yaml         # Scale points and resource limits
│
├── scripts/                     # Entry points — each phase is a script
│   ├── 00_setup_environment.sh  # Install deps, verify env
│   ├── 01_download_dataset.py   # Download LMSYS-Chat-1M from HuggingFace
│   ├── 02_generate_embeddings.py # Embed queries with all-MiniLM-L6-v2
│   ├── 03_run_index_benchmark.py # Experiment 1: index comparison
│   ├── 04_run_cost_model.py     # Experiment 2: cost-based routing
│   ├── 05_run_eviction.py       # Experiment 3: eviction policies
│   ├── 06_run_scalability.py    # Experiment 4: scalability analysis
│   ├── 07_llm_judge.py          # Semantic validity check (500 pairs)
│   └── 08_generate_figures.py   # All plots and tables for the paper
│
├── slurm/                       # SLURM job scripts for Great Lakes
│   ├── embed.sbatch             # GPU job for embedding generation
│   ├── benchmark.sbatch         # CPU job for index benchmarks
│   └── scalability.sbatch       # Multi-node scalability runs
│
├── src/                         # Core library code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load LMSYS-Chat-1M, extract user queries
│   │   ├── preprocessor.py      # Clean, deduplicate, tokenize queries
│   │   └── sampler.py           # Create subsets: 10K, 50K, 100K, 500K, 1M
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── encoder.py           # Wrapper around sentence-transformers
│   │
│   ├── indexes/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract index interface
│   │   ├── flat_index.py        # FAISS Flat (brute-force, ground truth)
│   │   ├── hnsw_index.py        # FAISS HNSW with tunable M, efConstruction, efSearch
│   │   ├── ivf_index.py         # FAISS IVF with tunable nlist, nprobe
│   │   ├── lsh_index.py         # FAISS LSH with tunable nbits
│   │   └── factory.py           # Factory: config → index instance
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── semantic_cache.py    # Full semantic cache: embed → search → return/miss
│   │   ├── eviction/
│   │   │   ├── __init__.py
│   │   │   ├── lru.py           # LRU baseline
│   │   │   ├── lfu.py           # LFU baseline
│   │   │   ├── semantic.py      # Our semantic-aware eviction policy
│   │   │   └── oracle.py        # Oracle policy (knows future queries, upper bound)
│   │   └── cost_model.py        # Cost-based query routing: cache vs direct LLM
│   │
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Latency, throughput, recall@k, memory, build time
│   │   ├── runner.py            # Run a single benchmark experiment
│   │   ├── workload.py          # Generate realistic query streams (clustered, skewed, bursty)
│   │   └── profiler.py          # Memory and time profiling utilities
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── llm_judge.py         # LLM-as-judge for semantic validity (500 pairs)
│       └── analysis.py          # Statistical analysis, crossover detection
│
├── notebooks/                   # Jupyter notebooks for exploration (optional)
│   └── eda.ipynb                # Dataset EDA, embedding distribution analysis
│
├── results/                     # All outputs go here (gitignored except summaries)
│   ├── embeddings/              # Saved .npy embedding arrays
│   ├── benchmarks/              # JSON/CSV benchmark results
│   ├── figures/                 # Generated plots (PDF + PNG)
│   └── tables/                  # LaTeX tables for the paper
│
├── tests/                       # Unit and integration tests
│   ├── test_indexes.py          # Verify each index returns correct results on small data
│   ├── test_eviction.py         # Verify eviction correctness
│   ├── test_cost_model.py       # Verify routing decisions
│   └── test_cache.py            # End-to-end cache test
│
└── paper/                       # LaTeX paper (if writing in LaTeX)
    └── figures/                 # Symlink or copy from results/figures/
```

---

## Dependencies

```
# Core
faiss-cpu>=1.7.4          # Use faiss-gpu on Great Lakes
numpy>=1.24
pandas>=2.0
sentence-transformers>=2.2
torch>=2.0
transformers>=4.30
datasets>=2.14            # HuggingFace datasets for LMSYS-Chat-1M

# Benchmarking & Profiling
psutil>=5.9
memory-profiler>=0.61
tqdm>=4.65

# Cost Model & Analysis
scipy>=1.11
scikit-learn>=1.3

# LLM-as-Judge
openai>=1.0               # Or anthropic SDK — for semantic validity check

# Visualization
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15              # Interactive plots (optional)

# Config & Utilities
pyyaml>=6.0
click>=8.1                # CLI interface
python-dotenv>=1.0

# Testing
pytest>=7.4
```

---

## Phase-by-Phase Execution Plan

---

### PHASE 0: Project Setup

**Where:** Local (MacBook)
**Goal:** Initialize repo, install deps, verify FAISS works

**Steps:**
1. Create the full directory structure shown above.
2. Initialize `pyproject.toml` with all dependencies.
3. Verify FAISS installation:
   ```python
   import faiss
   d = 384
   index = faiss.IndexFlatL2(d)
   xb = np.random.rand(1000, d).astype('float32')
   index.add(xb)
   D, I = index.search(xb[:5], k=10)
   assert I.shape == (5, 10)
   print("FAISS OK")
   ```
4. Verify sentence-transformers installation:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   emb = model.encode(["test query"])
   assert emb.shape == (1, 384)
   print("Encoder OK")
   ```
5. Write a basic `README.md` with setup instructions.
6. Create `.env.example` with `OPENAI_API_KEY=your-key-here`.

**Validation:** All imports succeed, FAISS search returns correct neighbors on toy data.

---

### PHASE 1: Data Acquisition & Preprocessing

**Where:** Local for download + small subset; Great Lakes for full embedding generation
**Goal:** Download LMSYS-Chat-1M, extract user queries, generate embeddings

**Step 1.1: Download Dataset**
```python
from datasets import load_dataset
ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
```
- The dataset is ~3.5GB. Download once and cache locally.
- Each row has a `conversation` field (list of turns). Extract only the **first user message** from each conversation — this represents the initial query which is what gets cached.

**Step 1.2: Preprocess Queries**
- Extract first user turn from each conversation.
- Filter: remove empty queries, queries shorter than 3 tokens, queries longer than 512 tokens.
- Deduplicate exact matches (keep count for frequency analysis).
- Store as `data/processed_queries.parquet` with columns: `[query_id, query_text, token_count, original_index]`.
- Expected output: ~800K–950K unique queries after filtering.

**Step 1.3: Generate Embeddings**

**On Great Lakes (GPU):**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
# Batch encode with batch_size=512 for GPU efficiency
embeddings = model.encode(queries, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
np.save('results/embeddings/full_embeddings.npy', embeddings.astype('float32'))
```
- Estimated time on A100: ~30-45 min for 1M queries
- Normalize embeddings (L2 norm = 1) so cosine similarity = dot product = 1 - L2²/2
- Save as float32 `.npy` file (~1.5GB for 1M × 384)

**On Local (small subset for dev):**
- Generate embeddings for 10K and 50K subsets on CPU (~5-15 min)
- Use these for all local development and debugging

**Step 1.4: Create Scale Subsets**
- From the full embedding set, create deterministic subsets using fixed random seeds:
  - `10k_subset` — 10,000 vectors (development, unit tests)
  - `50k_subset` — 50,000 vectors (local benchmarks)
  - `100k_subset` — 100,000 vectors (production-scale benchmark)
  - `500k_subset` — 500,000 vectors (high-traffic benchmark)
  - `full` — all vectors (enterprise-scale benchmark)
- Save each as a separate `.npy` file in `results/embeddings/`
- Also save corresponding query text subsets for the LLM-as-judge phase.

**Step 1.5: Dataset EDA (optional but recommended)**
- Plot embedding dimension distribution, cosine similarity distribution of random pairs
- Cluster embeddings (k-means, k=50) to validate that LLM queries are indeed clustered
- Plot query length distribution, identify topic clusters
- Save plots to `results/figures/eda/`

**Validation:**
- Embedding shape matches `(N, 384)` and dtype is `float32`
- Embeddings are L2-normalized: `np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)`
- Cosine similarity between "What is photosynthesis?" and "How does photosynthesis work?" > 0.8
- Cosine similarity between "What is photosynthesis?" and "Best pizza in NYC?" < 0.3

**SLURM Script Template (`slurm/embed.sbatch`):**
```bash
#!/bin/bash
#SBATCH --job-name=embed_gen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cse584w25_class
#SBATCH --output=logs/embed_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate

python scripts/02_generate_embeddings.py --device cuda --batch-size 512 --output results/embeddings/
```

> **IMPORTANT:** Check the correct SLURM account name and partition with `sacctmgr show assoc user=$USER`. The account above is a placeholder — Yash should confirm the actual account string.

---

### PHASE 2: Index Benchmarking (Experiment 1)

**Where:** Local for development/debugging on 10K–50K; Great Lakes for official runs at all scales
**Goal:** Compare HNSW, IVF, LSH, Flat across latency, throughput, recall@k, memory, build time

**Step 2.1: Implement Index Wrappers**

Each index wrapper (`src/indexes/`) must implement this interface:
```python
class BaseIndex(ABC):
    def build(self, vectors: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...
    def add(self, vectors: np.ndarray) -> None: ...
    def remove(self, ids: np.ndarray) -> None: ...  # needed for eviction
    @property
    def memory_usage_bytes(self) -> int: ...
    @property
    def ntotal(self) -> int: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**Index Configurations to Test:**

| Index | Parameters | Notes |
|-------|-----------|-------|
| **Flat** | (none) | Brute-force, exact, ground truth |
| **HNSW** | M ∈ {16, 32, 64}, efConstruction ∈ {64, 128, 256}, efSearch ∈ {32, 64, 128, 256} | Best for low-latency, high-recall |
| **IVF** | nlist ∈ {64, 256, 1024, 4096}, nprobe ∈ {1, 4, 16, 64, 128} | Best for large-scale with memory constraints |
| **LSH** | nbits ∈ {384, 768, 1536} | Baseline approximate method |

Store these in `configs/index_benchmark.yaml`.

**Step 2.2: Implement Benchmark Runner**

For each (index_type, parameter_set, dataset_size) combination:
1. **Build Phase:** Build index from database vectors. Measure wall-clock build time and peak memory.
2. **Search Phase:** Run 1,000 queries (held-out from the database). For each query, measure:
   - **Latency:** Time per individual search (use `time.perf_counter_ns()`)
   - **Throughput:** Batch search time for 1000 queries / 1000
   - **Recall@k:** Compare returned IDs to Flat (ground truth) results. Recall@k = |intersection(approx_top_k, exact_top_k)| / k
   - **Memory:** `index.memory_usage_bytes` or `psutil` process memory delta
3. Repeat each measurement 3 times, report mean and std.

**Step 2.3: Workload Generation**

Implement realistic LLM workloads in `src/benchmark/workload.py`:
- **Uniform:** Random query selection (baseline)
- **Clustered:** 80% of queries come from 20% of topic clusters (Zipf distribution over k-means clusters)
- **Bursty:** Temporal locality — queries arrive in bursts of similar topics (simulate with sliding window over clustered queries)
- **Mixed:** Combination of above

**Step 2.4: Run Benchmarks**

```bash
# Local dev (10K, quick sanity check)
python scripts/03_run_index_benchmark.py --size 10k --output results/benchmarks/

# Great Lakes (all sizes, all configs)
sbatch slurm/benchmark.sbatch
```

**Expected Key Results:**
- Flat: exact recall, high latency (~45ms at 100K, ~500ms at 1M)
- HNSW (M=32, ef=128): ~0.5ms latency, >0.95 recall at 100K
- IVF (nlist=1024, nprobe=16): ~1-2ms latency, >0.90 recall at 100K
- LSH: fastest build, lowest recall (~0.7-0.8)

**Output Format:** Save as `results/benchmarks/index_benchmark_{size}.json`:
```json
{
  "index_type": "hnsw",
  "params": {"M": 32, "efConstruction": 128, "efSearch": 128},
  "dataset_size": 100000,
  "build_time_s": 12.5,
  "memory_mb": 245.3,
  "search_latency_ms": {"mean": 0.52, "std": 0.03, "p50": 0.48, "p95": 0.71, "p99": 1.1},
  "throughput_qps": 1923,
  "recall_at_1": 0.97,
  "recall_at_5": 0.95,
  "recall_at_10": 0.93,
  "workload": "clustered"
}
```

**Validation:**
- Flat index recall@k = 1.0 always (it IS ground truth)
- HNSW recall@10 > 0.90 with default params
- Latency measurements are stable (std < 20% of mean)
- Memory usage grows sub-linearly or linearly with dataset size

---

### PHASE 3: Cost-Based Query Routing (Experiment 2)

**Where:** Mostly local (this is computation-light, involves modeling)
**Goal:** Build a cost model that decides cache lookup vs. direct LLM call

**Step 3.1: Define Cost Model**

```
Cost(cache)  = T_embed + T_search(index, N) + P(miss) × T_llm
Cost(direct) = T_llm

Use cache when: Cost(cache) < Cost(direct)
→ T_embed + T_search(index, N) < (1 - P(miss)) × T_llm
→ T_embed + T_search(index, N) < P(hit) × T_llm
```

Components to estimate:
- `T_embed`: Embedding generation time. Measure empirically for `all-MiniLM-L6-v2` by query length.
- `T_search(index, N)`: Search latency as a function of index type and cache size N. Use Phase 2 results.
- `P(miss)`: Cache miss probability. Estimate from:
  - Cache size relative to query universe
  - Similarity threshold setting
  - Query distribution (clustered queries → higher hit rate)
- `T_llm`: LLM API latency. Use 800ms as default (configurable). Also model token-dependent latency: `T_llm = T_base + T_per_token × output_tokens`.

**Step 3.2: Implement Router**

```python
class QueryRouter:
    def should_use_cache(self, query: str, cache_state: CacheState) -> bool:
        t_embed = self.estimate_embed_time(query)
        t_search = self.estimate_search_time(cache_state.index_type, cache_state.size)
        p_hit = self.estimate_hit_probability(query, cache_state)
        t_llm = self.estimate_llm_time(query)

        cost_cache = t_embed + t_search + (1 - p_hit) * t_llm
        cost_direct = t_llm
        return cost_cache < cost_direct
```

**Step 3.3: Hit Probability Estimation**

Implement multiple estimators and compare:
1. **Global frequency:** P(hit) = cache_size / estimated_query_universe_size
2. **Cluster-based:** If query is near a dense cluster in embedding space, P(hit) is higher
3. **Sliding window:** Track recent hit rate over last N queries
4. **Hybrid:** Combine above with learned weights

**Step 3.4: Evaluation**

- Simulate a query stream (10K queries) against caches of various sizes.
- For each query, record: router decision, actual outcome (hit/miss), time saved/wasted.
- Metrics:
  - **Routing accuracy:** % of correct decisions (compared to oracle that knows the outcome)
  - **Net latency savings:** Total time saved by correct cache hits minus time wasted on misses
  - **Break-even analysis:** At what hit rate does caching become beneficial for each index type?

- Baseline: **Always-cache** policy (always attempt cache lookup)
- Compare: Our cost-based router vs. always-cache vs. never-cache

**Output:** `results/benchmarks/cost_model_results.json` with routing accuracy, latency savings, and break-even points per index type.

**Validation:**
- When cache is empty (size=0), router should say "don't use cache" 100% of the time
- When hit rate is very high (>0.95), router should say "use cache" ~100% of the time
- Router should avoid cache for very short queries (embedding overhead dominates)

---

### PHASE 4: Semantic-Aware Eviction (Experiment 3)

**Where:** Local for development on 10K–50K; Great Lakes for 100K+ runs
**Goal:** Implement and evaluate semantic-aware eviction vs. LRU, LFU, and oracle

**Step 4.1: Implement Eviction Policies**

**LRU (baseline):** Standard least-recently-used. Use `OrderedDict` or doubly-linked list + hash map.

**LFU (baseline):** Least-frequently-used. Track access counts per entry.

**Semantic-Aware (our contribution):**
```python
def compute_eviction_score(entry, cache_entries, similarity_threshold=0.85):
    """
    Score = redundancy_score / (recency_score + frequency_score)
    High score → evict first

    redundancy_score: How many other entries are similar to this one
    recency_score: Normalized recency (1 = most recent, 0 = oldest)
    frequency_score: Normalized access frequency
    """
    neighbors = count_neighbors_above_threshold(entry.embedding, cache_entries, similarity_threshold)
    redundancy = neighbors / len(cache_entries)
    recency = normalize_recency(entry.last_access_time)
    frequency = normalize_frequency(entry.access_count)

    score = redundancy / (recency + frequency + epsilon)
    return score
```

Key insight: If an entry has many near-duplicates in the cache, evicting it loses little coverage. If an entry is unique (no near neighbors), it's irreplaceable and should be kept.

**Oracle (upper bound):** Bélády's optimal algorithm — evict the entry that won't be used for the longest time in the future. Requires knowledge of future queries (offline only, serves as theoretical upper bound).

**Step 4.2: Efficient Redundancy Computation**

Computing pairwise similarities for every eviction is expensive. Optimization strategies:
1. **Batch updates:** Recompute redundancy scores every K evictions (not every one).
2. **Approximate neighbors:** Use HNSW index over cache entries for fast neighbor counting (don't do brute-force).
3. **Incremental updates:** When adding/removing entries, only update scores for affected neighbors.

Choose batch updates with K=100 as the default. This amortizes the cost while keeping scores fresh.

**Step 4.3: Experimental Design**

- **Cache size:** Fix at 10%, 20%, 30% of the total unique query count.
- **Query stream:** Use temporal ordering from LMSYS-Chat-1M (conversations are timestamped).
- **Warm-up:** Fill cache with first N queries (no eviction during warm-up).
- **Measurement:** After warm-up, measure hit rate over the next 50K queries.
- **Repeat:** 3 runs with different random seeds for stream shuffling variants.

**Step 4.4: Metrics**
- **Hit rate:** % of queries answered from cache (primary metric)
- **Semantic coverage:** Average minimum distance from any uncached query to its nearest cached entry (lower = better coverage)
- **Eviction overhead:** Time spent computing eviction scores per eviction event

**Output:** `results/benchmarks/eviction_{policy}_{cache_pct}.json`

**Validation:**
- Oracle always achieves highest hit rate (it's optimal)
- Semantic-aware should beat LRU/LFU on workloads with clustered queries
- All policies converge to similar performance on uniform random workloads (no structure to exploit)
- Eviction overhead for semantic-aware is <10ms per eviction event on 50K cache

---

### PHASE 5: Scalability Analysis (Experiment 4)

**Where:** Great Lakes (need consistent hardware for fair comparison)
**Goal:** Measure how all metrics scale from 10K to 1M and identify crossover points

**Step 5.1: Experimental Matrix**

For each scale point {10K, 50K, 100K, 250K, 500K, 1M}:
- Run all 4 index types with their best parameter configurations (from Phase 2)
- Measure: build time, memory, search latency, throughput, recall@10
- Run best eviction policy at 20% cache size
- Run cost model and record routing decisions

**Step 5.2: Crossover Detection**

Automatically detect crossover points where one index becomes better than another:
```python
def find_crossover(metric_A, metric_B, sizes):
    """Find the dataset size where metric_A crosses metric_B."""
    diff = metric_A - metric_B
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    # Interpolate to find exact crossover point
    ...
```

Expected crossovers:
- Flat → HNSW: ~5K–10K entries (Flat is faster below this due to no build overhead)
- HNSW → IVF: ~200K–500K entries (IVF becomes more memory-efficient at scale)

**Step 5.3: Resource Scaling Models**

Fit scaling models to empirical data:
- Memory: `M(n) = a * n + b` (linear) or `M(n) = a * n * log(n) + b` (for HNSW)
- Build time: `T(n) = a * n * log(n) + b` (for HNSW) or `T(n) = a * n + b` (for IVF)
- Search latency: `T(n) = a * log(n) + b` (for HNSW) or `T(n) = a * sqrt(n) + b` (for IVF)

Use `scipy.optimize.curve_fit` to find parameters and R² goodness of fit.

**Output:** `results/benchmarks/scalability_results.json` with all metrics at all scales, crossover points, and fitted model parameters.

**Validation:**
- Flat latency grows linearly with N
- HNSW latency grows roughly logarithmically
- IVF latency depends mainly on nprobe, grows with sqrt(nlist)
- Memory for all indexes grows at least linearly with N

---

### PHASE 6: LLM-as-Judge Semantic Validity (Experiment 5)

**Where:** Local (only 500 API calls)
**Goal:** Verify that high-similarity cache hits return semantically valid responses

**Step 6.1: Sample Selection**

From Phase 2 results, select 500 query pairs where:
- 200 pairs with similarity > 0.95 (very high — should almost all be valid)
- 150 pairs with similarity 0.85–0.95 (medium — some might be invalid)
- 100 pairs with similarity 0.75–0.85 (borderline — expect some failures)
- 50 pairs with similarity 0.60–0.75 (low — expect many failures)

For each pair, we have:
- `query_original`: The query that was cached
- `response_cached`: The response stored in cache (use first assistant turn from LMSYS data)
- `query_new`: The new query that matched via embedding similarity

**Step 6.2: LLM-as-Judge Prompt**

```
You are evaluating whether a cached response is semantically valid for a new query.

Original query: {query_original}
Cached response: {response_cached}
New query: {query_new}
Cosine similarity: {similarity_score}

Would the cached response adequately answer the new query?
Respond with ONLY one of: VALID, PARTIALLY_VALID, INVALID

VALID: The response fully answers the new query.
PARTIALLY_VALID: The response is related but incomplete or slightly off.
INVALID: The response does not answer the new query.
```

Use Claude (via Anthropic API) or GPT-4o-mini for judging. Budget: ~$2-5 for 500 calls.

**Step 6.3: Analysis**

- Compute validity rate at each similarity threshold
- Plot: x=similarity threshold, y=% valid responses
- Identify the optimal similarity threshold where validity drops below 90%
- This directly informs what threshold semantic caches should use

**Output:** `results/benchmarks/llm_judge_results.json` and `results/figures/validity_vs_threshold.pdf`

**Validation:**
- Pairs with similarity > 0.95 should be >90% VALID
- Pairs with similarity < 0.70 should be <50% VALID
- The curve should be monotonically increasing (higher similarity → more valid)

---

### PHASE 7: Visualization & Report Generation

**Where:** Local
**Goal:** Generate all figures and tables for the final paper/report

**Required Figures:**

1. **Index Latency Comparison** (bar chart or violin plot)
   - X: index type, Y: search latency (ms), grouped by dataset size
   - Show median, p95, p99

2. **Recall vs. Latency Tradeoff** (scatter/Pareto plot)
   - Each point = (latency, recall@10) for one index configuration
   - Color by index type, annotate Pareto-optimal configs

3. **Scalability Curves** (line plot)
   - X: dataset size (log scale), Y: latency/memory/recall
   - One line per index type
   - Annotate crossover points with vertical dashed lines

4. **Cost Model Decision Boundary**
   - X: cache size, Y: estimated hit probability
   - Shaded regions for "use cache" vs "call LLM" per index type

5. **Eviction Policy Hit Rate** (line plot)
   - X: number of queries processed, Y: cumulative hit rate
   - One line per eviction policy
   - Include oracle as upper bound (dashed line)

6. **Semantic Coverage Visualization**
   - 2D t-SNE/UMAP of embeddings, colored by whether cached or evicted
   - Show that semantic-aware eviction maintains better coverage

7. **Validity vs. Similarity Threshold** (line plot with confidence interval)
   - X: cosine similarity, Y: % valid (from LLM judge)
   - Shade 95% CI

8. **Build Time vs. Dataset Size** (line plot)
   - X: dataset size, Y: build time
   - One line per index type

**Required Tables:**

1. **Main Results Table:** Index type | Params | Latency (ms) | Recall@10 | Memory (MB) | Build Time (s) — at 100K entries
2. **Eviction Results Table:** Policy | Hit Rate (%) at 10%, 20%, 30% cache sizes
3. **Cost Model Results Table:** Router | Routing Accuracy | Net Latency Savings
4. **Crossover Points Table:** Transition | Dataset Size | Recommended Action

**Style:**
- Use `matplotlib` with `seaborn` styling (`sns.set_theme(style="whitegrid")`)
- Font size ≥ 12 for axis labels, ≥ 10 for tick labels
- Save as both PDF (for paper) and PNG (for presentation)
- Use colorblind-friendly palette (`sns.color_palette("colorblind")`)

---

## Key Implementation Notes

### FAISS Index Construction Patterns

```python
# Flat (exact)
index = faiss.IndexFlatL2(384)

# HNSW
index = faiss.IndexHNSWFlat(384, M)  # M = number of connections
index.hnsw.efConstruction = efConstruction
index.hnsw.efSearch = efSearch

# IVF
quantizer = faiss.IndexFlatL2(384)
index = faiss.IndexIVFFlat(quantizer, 384, nlist)
index.train(training_vectors)  # MUST train before adding
index.nprobe = nprobe

# LSH
index = faiss.IndexLSH(384, nbits)
```

### FAISS Gotchas
- **IVF requires training** on a representative sample before adding vectors. Use at least `64 * nlist` training vectors.
- **HNSW does not support removal.** For eviction experiments with HNSW, use `IndexIDMap` wrapper or rebuild the index periodically.
- **LSH in FAISS is not production-grade.** Expect lower recall than HNSW/IVF. This is fine — we're documenting it.
- **Normalize embeddings** if using inner product (cosine) similarity. Since we use L2 distance on normalized vectors, cosine sim = 1 - L2²/2.
- **FAISS search returns L2 distances**, not similarities. Convert: `similarity = 1 - distance / 2` (for normalized vectors).

### Timing Best Practices
- Use `time.perf_counter_ns()` for nanosecond precision on individual queries.
- Warm up the index with 100 dummy queries before timing (JIT/cache effects).
- Run each measurement 3x and report mean ± std.
- For throughput, measure batch search time and divide by batch size.

### Memory Measurement
- FAISS: Use `faiss.vector_to_array(index.get_xb()).nbytes` for raw vector storage.
- Process-level: Use `psutil.Process().memory_info().rss` before and after index construction.
- Report both: FAISS-internal and total process memory delta.

### Reproducibility
- Fix all random seeds: `np.random.seed(42)`, `faiss.seed(42)`.
- Record exact package versions in `requirements.txt` or `pyproject.toml` lock file.
- Record hardware info: CPU model, RAM, OS version, FAISS version.
- All experiments must be runnable from a single command per phase.

---

## Great Lakes HPC Quick Reference

```bash
# Connect
ssh uniqname@greatlakes.arc-ts.umich.edu

# Check allocation
my_accounts

# Module setup
module load python/3.11 cuda/12.1

# Create venv (do this once)
python -m venv ~/cse584-venv
source ~/cse584-venv/bin/activate
pip install faiss-gpu numpy pandas sentence-transformers torch datasets

# Submit job
sbatch slurm/embed.sbatch

# Check job status
squeue -u $USER

# Interactive GPU session (for debugging)
salloc --partition=gpu --gpus=1 --mem=32G --time=01:00:00 --account=cse584w25_class

# Data storage
# Use /scratch/cse584w25_class_root/cse584w25_class/ for large files
# Home directory has limited quota (~80GB)
```

---

## CLI Interface

All scripts should be runnable via CLI with `click`:

```bash
# Phase 0
python scripts/00_setup_environment.sh

# Phase 1
python scripts/01_download_dataset.py --output data/
python scripts/02_generate_embeddings.py --input data/processed_queries.parquet --output results/embeddings/ --device cpu --batch-size 64

# Phase 2
python scripts/03_run_index_benchmark.py --size 10k --config configs/index_benchmark.yaml --output results/benchmarks/

# Phase 3
python scripts/04_run_cost_model.py --embeddings results/embeddings/100k_subset.npy --output results/benchmarks/

# Phase 4
python scripts/05_run_eviction.py --policy semantic --cache-pct 0.2 --embeddings results/embeddings/100k_subset.npy --output results/benchmarks/

# Phase 5
python scripts/06_run_scalability.py --config configs/scalability.yaml --embeddings-dir results/embeddings/ --output results/benchmarks/

# Phase 6
python scripts/07_llm_judge.py --pairs results/benchmarks/judge_pairs.json --model claude-sonnet-4-20250514 --output results/benchmarks/

# Phase 7
python scripts/08_generate_figures.py --results-dir results/benchmarks/ --output results/figures/
```

---

## Testing Strategy

Run tests before each phase to catch bugs early:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_indexes.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Key test cases:**
- `test_indexes.py`: Each index returns correct results on 100-vector toy data. Recall@10 = 1.0 for Flat. Recall@10 > 0.8 for all approximate indexes on easy data.
- `test_eviction.py`: LRU evicts oldest entry. LFU evicts least accessed. Semantic evicts most redundant. Cache size never exceeds limit.
- `test_cost_model.py`: Router returns "skip cache" when cache is empty. Router returns "use cache" when P(hit) is very high.
- `test_cache.py`: End-to-end test — add entries, query, verify hits and misses.

---

## Error Handling & Edge Cases

- **Out of memory:** If FAISS index exceeds available RAM, fail gracefully with a clear error message. For IVF at 1M scale, use `IndexIVFFlat` not `IndexIVFPQ` (we want exact distances within clusters).
- **Empty cache:** Cost model should return "skip cache" immediately.
- **Duplicate queries:** Handle gracefully — don't add to cache if exact embedding already exists.
- **FAISS thread safety:** FAISS search is thread-safe for reading but not for writing. Use a lock for add/remove operations if parallelizing.
- **NaN embeddings:** Check for NaN in embeddings after generation. Sentence-transformers can produce NaN for extremely long or malformed inputs.

---

## Success Criteria

The project is complete when:

1. ✅ All 4 index types benchmarked at 5+ scale points with latency, recall, memory, throughput
2. ✅ Cost model demonstrates >10% latency savings over always-cache baseline
3. ✅ Semantic-aware eviction achieves >5% hit rate improvement over LRU on clustered workloads
4. ✅ Crossover points identified (e.g., "switch from HNSW to IVF at N=X")
5. ✅ LLM-as-judge validates semantic threshold selection
6. ✅ All figures and tables generated for the paper
7. ✅ Code is reproducible: `git clone` → install deps → run scripts → get same results
