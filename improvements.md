# CLEVER Improvements Backlog

This document captures improvements identified from a codebase review.  
Per request, this is documentation-only (no code edits).

## P0 — Correctness and project consistency (highest priority)

### 1) Wire benchmark workloads into actual runs
- **Observed in:** `src/benchmark/runner.py`, `configs/index_benchmark.yaml`
- **Issue:** `BenchmarkRunner` loads `workloads` and imports `generate_workload`, but workload generation is never used in execution.
- **Why it matters:** Reported results are effectively one workload despite config implying multiple workload types.
- **Improve by:**
  - Running search/evaluation per workload (`uniform`, `clustered`, `bursty`).
  - Persisting `workload` in each result row.
  - Validating expected number of results = `index_configs × params × workloads`.

### 2) Fix cache-vs-LLM miss-path latency math in cost savings
- **Observed in:** `src/router/cost_model.py`
- **Issue:** `cost_savings()` computes miss-path latency as only `llm_latency_ms`; it omits cache lookup latency paid before a miss.
- **Why it matters:** It overestimates latency savings and can bias routing conclusions.
- **Improve by:** using:
  - `actual_latency = hit_rate * cache_latency + (1 - hit_rate) * (cache_latency + llm_latency)`
  - keep monetary model explicit and separate from latency.

### 3) Define and enforce true cache-hit semantics in `SemanticCache`
- **Observed in:** `src/cache/semantic_cache.py`
- **Issue:** `lookup()` marks `hit=True` whenever a neighbor index is valid; no similarity threshold is applied. `_n_hits` is never incremented.
- **Why it matters:** Stats and evaluation can misrepresent real cache usefulness.
- **Improve by:**
  - introducing threshold-aware hit definition (or returning “match found” vs “cache hit” separately),
  - incrementing `_n_hits` only for true hits,
  - exposing threshold in lookup APIs and reporting.

### 4) Align README phase scripts with implemented scripts
- **Observed in:** `README.md` vs `scripts/`
- **Issue:** README references scripts not present (`04_run_cost_model.py`, `05_run_eviction.py`, `06_run_scalability.py`, `07_llm_judge.py`, `08_generate_figures.py`), while existing scripts are named differently (`06_run_routing_eval.py`, visualization scripts, synthetic data generator).
- **Why it matters:** Repro steps are currently misleading.
- **Improve by:** either adding missing scripts or updating README/phase table to the actual runnable pipeline.

### 5) Resolve eviction phase gap (implemented vs promised)
- **Observed in:** `src/cache/eviction/` (only `__init__.py`)
- **Issue:** LRU/LFU/semantic/oracle eviction implementations are absent despite being core project claims.
- **Why it matters:** Major planned experiment is currently non-executable.
- **Improve by:** adding eviction policy modules and integrating them into cache + benchmark scripts, or explicitly de-scoping in docs.

## P1 — Reliability and evaluation quality

### 6) Expand tests beyond environment/setup checks
- **Observed in:** `tests/test_setup.py` (single file, mostly dependency/model checks)
- **Issue:** Core logic (indexes wrapper behavior, cache semantics, routing math, benchmark runner behavior) lacks targeted unit/integration tests.
- **Improve by adding tests for:**
  - `BenchmarkRunner`: workload application and result cardinality.
  - `CostModel`: miss-path equation and break-even behavior.
  - `SemanticCache`: insert/lookup stats and threshold hit handling.
  - Index wrappers: add/remove contracts, serialization, IVF `nlist` edge cases.

### 7) Stabilize test/runtime compatibility
- **Observed during test run:** `pytest` abort with Python 3.13 while importing `torch/transformers/sentence-transformers`.
- **Issue:** Runtime combination appears unstable for current dependency set.
- **Improve by:**
  - pinning a validated Python version (e.g., 3.11/3.12) in docs/CI,
  - validating dependency lock against that interpreter,
  - marking model-heavy tests as slow/integration and optional in default CI.

### 8) Make routing quality metric more faithful
- **Observed in:** `src/evaluation/routing_evaluator.py`, `src/router/adaptive_router.py`
- **Issue:** quality proxy uses distance inequality (`cache_dist <= gt_dist * 2 + 0.01`), which is weakly tied to semantic answer validity.
- **Improve by:**
  - also computing ID-overlap/recall@k for retrieval quality,
  - reporting threshold-vs-validity from judged pairs where available,
  - separating “retrieval quality” and “response validity” metrics.

### 9) Improve benchmark statistical rigor
- **Observed in:** `src/benchmark/runner.py`, `src/benchmark/profiler.py`
- **Issue:** per-query latency distribution is measured, but build/memory are effectively single-shot per config and workload dimension is missing.
- **Improve by:**
  - repeating full runs per config/workload with independent seeds,
  - reporting mean/std/CI for build and memory too,
  - recording seed + hardware metadata in output JSON.

## P2 — Maintainability and UX improvements

### 10) Data preprocessing robustness
- **Observed in:** `src/data/preprocessor.py`, `src/data/loader.py`
- **Issue:** token counting is whitespace-based (not model-token-based); metadata field naming (`num_turns` from `turn`) may not reflect true turns depending on dataset schema.
- **Improve by:**
  - documenting this approximation clearly,
  - optionally adding tokenizer-based counting mode,
  - validating source schema fields at load time.

### 11) Strengthen package/reproducibility metadata
- **Observed in:** `pyproject.toml`, `requirements.txt`
- **Issue:** project has both floating constraints and hard-pinned requirements; reproducibility expectations are unclear.
- **Improve by:**
  - choosing one primary install path (locked vs flexible),
  - documenting supported matrix (OS/Python/FAISS/Torch),
  - adding a simple CI smoke benchmark on a tiny subset.

### 12) Improve user-facing run manifests
- **Observed across scripts**
- **Issue:** outputs are generated, but experiment provenance (config hash, git SHA, seed, command) is not consistently captured.
- **Improve by:** writing a `run_manifest.json` alongside each benchmark output.

## Suggested execution order
1. **P0 items 1–5** (correctness + consistency)  
2. **P1 items 6–9** (reliability + evaluation credibility)  
3. **P2 items 10–12** (long-term maintainability/reproducibility)

## Quick acceptance checklist
- [ ] Workload-specific benchmark results are generated and labeled.
- [ ] Cost model miss-path equation includes cache lookup overhead.
- [ ] Cache hit rate semantics are threshold-correct and stats reflect true hits.
- [ ] README scripts exactly match runnable scripts in `scripts/`.
- [ ] Eviction phase is either implemented end-to-end or explicitly removed from claims.
- [ ] Core logic tests exist and pass on a pinned supported Python version.
