# CLEVER — Top-Journal Readiness Report (Completed Workstreams Only)

This report is a **publication-quality upgrade plan** for the workstreams already implemented in this repo.
It focuses only on currently completed phases/components:

- Environment + setup pipeline
- Data extraction/preprocessing + embeddings
- Index benchmarking
- Cost-based routing
- Eviction policy evaluation
- Current testing, reproducibility, and result packaging

> **Out of scope in this document:** not-yet-completed future phases (e.g., new scalability phase tooling, separate LLM-judge phase expansion).

---

## 1) Current maturity snapshot

## What is already strong
- Good modular design (`src/indexes`, `src/cache`, `src/router`, `src/benchmark`).
- Clear abstraction boundaries (`BaseIndex`, `EvictionPolicy`, `SemanticCache`).
- Strong eviction policy implementation breadth (LRU/LFU/Semantic/Oracle) and dedicated tests.
- End-to-end runnable scripts and SLURM pipeline coverage.
- Rich figure generation already present.

## What prevents top-journal readiness today
- Some **logic risks** can bias conclusions.
- Some **evaluation protocols** are not strict enough for publication claims.
- Existing saved artifacts are **incomplete/inconsistent** with intended experiment matrix.
- Statistical and reproducibility metadata are not yet at journal-grade rigor.

---

## 2) P0 — Publication blockers (must fix first)

### P0.1 Eliminate workload leakage in index benchmarks
- **Observed in:** `src/benchmark/runner.py`
- **Issue:** For non-uniform workloads, query vectors are sampled from `db_vectors` instead of a held-out query pool.
- **Why this is critical:** It creates exact/self-neighbor leakage and inflates recall/latency claims.
- **Code/logic changes:**
  - Generate workload structure over held-out queries only.
  - Keep strict DB/query disjointness for *all* workload types.
  - Add assertion that query indices are never from DB IDs.
- **Acceptance criteria:**
  - No overlap between DB and query IDs in any workload.
  - Recall curves degrade realistically vs. current inflated values.

### P0.2 Enforce workload-complete benchmark outputs
- **Observed in:** `results/benchmarks/index_benchmark_*.json`
- **Issue:** Current saved benchmark files are flat lists with 22 rows each (no workload field), despite workload config support.
- **Why this is critical:** The reported evidence does not match configured experiment design.
- **Code/logic changes:**
  - Hard-validate expected row count:
    `n_indexes × n_param_sets × n_workloads` (after legal skips).
  - Fail run if workload dimension is missing from output rows.
  - Persist manifest with workload list and cardinality checks.
- **Acceptance criteria:**
  - Every row has `workload`.
  - Cardinality check passes automatically.

### P0.3 Fix “full scale” labeling/data-size inconsistency
- **Observed in:** `results/benchmarks/index_benchmark_full.json` (`dataset_size` ~7.5K).
- **Issue:** `full` label currently maps to a small dataset in stored artifacts.
- **Why this is critical:** Mislabeling scale undermines trust in performance claims.
- **Code/logic changes:**
  - Add explicit `effective_n_vectors` and source file metadata to manifests.
  - Refuse writing `*_full*` outputs if input size is below configured minimum for “full.”
  - Introduce run-time warning/error for mislabeled scales.
- **Acceptance criteria:**
  - “full” outputs always correspond to true full-scale embeddings for that run context.

### P0.4 Make Oracle policy actually optimal in-stream
- **Observed in:** `src/cache/eviction/oracle.py` (`on_access` is `pass`).
- **Issue:** Oracle next-use state is not updated after accesses, so it may not remain a true Bélády upper bound over time.
- **Why this is critical:** Oracle is used as theoretical upper bound; if incorrect, all eviction comparisons are weakened.
- **Code/logic changes:**
  - Maintain per-entry queue of future access positions.
  - Pop consumed positions on access and advance to next true future use.
  - Add invariant test: oracle hit-rate must be >= every baseline under same stream/cache.
- **Acceptance criteria:**
  - Oracle consistently dominates LRU/LFU/Semantic across tested settings.

### P0.5 Prevent trivial adaptive-router optimization
- **Observed in:** `src/router/adaptive_router.py`, `src/evaluation/routing_evaluator.py`
- **Issue:** Current quality objective can still favor near-max thresholds leading to almost-always-cache behavior with optimistic savings.
- **Why this is critical:** Can overstate routing benefits and weaken causal interpretation.
- **Code/logic changes:**
  - Optimize on constrained objective:
    maximize latency/cost savings **subject to** strict retrieval-quality floor.
  - Add explicit penalty for false cache accepts (not only distance smoothness).
  - Report decision boundary and rejected-threshold region.
- **Acceptance criteria:**
  - Best threshold is stable across seeds and does not collapse to trivial max threshold unless data truly supports it.

### P0.6 Use realistic temporal evaluation splits for routing and eviction
- **Observed in:** `src/evaluation/routing_evaluator.py`, `scripts/08_run_eviction.py`
- **Issue:** Heavy random shuffling weakens realism for production cache behavior.
- **Why this is critical:** Temporal locality and concept drift are central to cache research claims.
- **Code/logic changes:**
  - Use chronological split mode as default for final paper runs.
  - Keep random split only as sensitivity/ablation.
  - Track and report split strategy in every result artifact.
- **Acceptance criteria:**
  - Final reported main tables use temporal split; random split appears only in ablation.

### P0.7 Complete eviction experiment matrix before claiming policy gains
- **Observed in:** `results/eviction/eviction_results.json` (currently only `oracle`, only `10%`, one seed).
- **Issue:** Stored evidence currently covers only a tiny subset of intended comparisons.
- **Why this is critical:** No credible policy ranking without full matrix.
- **Code/logic changes:**
  - Enforce full run matrix: policies × cache sizes × seeds.
  - Add run-level validation that required cells are present before plotting.
  - Fail visualization if matrix is incomplete (instead of silently plotting partials).
- **Acceptance criteria:**
  - Final artifact includes LRU/LFU/Semantic/Oracle at each configured cache size and seed.

---

## 3) P1 — High-priority scientific quality upgrades

### P1.1 Strengthen statistical rigor for all completed experiments
- **Observed in:** benchmark/routing/eviction scripts and outputs.
- **Required upgrades:**
  - Use repeated independent runs for build time, latency, memory, hit rate.
  - Report mean/std + 95% CI.
  - Add statistical significance tests for key pairwise claims (e.g., Semantic vs LRU).
  - Include effect sizes, not just p-values.
- **Deliverable:** machine-readable `summary_stats.json` per experiment family.

### P1.2 Add publication-grade run manifests everywhere
- **Observed in:** manifests are inconsistent across scripts.
- **Required upgrades:**
  - Standardized manifest fields:
    - git SHA, dirty flag
    - command-line invocation
    - config hash
    - seed(s)
    - host/CPU/RAM/OS
    - package versions
  - Attach manifest to each result file.
- **Deliverable:** deterministic provenance for every figure/table.

### P1.3 Expand test suite to cover scientific correctness, not just functionality
- **Observed in:** tests currently strong for eviction + setup, limited elsewhere.
- **Required upgrades:**
  - `BenchmarkRunner` tests for workload cardinality and no-leak split correctness.
  - `AdaptiveRouter` tests for constrained-threshold selection behavior.
  - `CostModel` regression tests on latency/cost equations.
  - “Oracle dominates” property tests across generated streams.
- **Deliverable:** targeted correctness tests guarding research claims.

### P1.4 Pin supported runtime and enforce it
- **Observed during execution:** Python 3.13 + torch/transformers instability in setup tests.
- **Required upgrades:**
  - Set official runtime to validated version (e.g., Python 3.11).
  - Add interpreter/version check in scripts and CI.
  - Mark heavy model tests separately (slow/integration).
- **Deliverable:** reproducible test pass/fail behavior.

### P1.5 Improve measurement fidelity for memory and timing
- **Observed in:** `src/benchmark/profiler.py`
- **Required upgrades:**
  - Distinguish peak RSS vs. post-build RSS.
  - Separate index memory from process noise where possible.
  - Store warmup strategy and timing mode in output metadata.
- **Deliverable:** defensible memory and latency methodology section.

---

## 4) P2 — Manuscript and artifact consistency upgrades

### P2.1 Align README claims with exactly runnable scripts/results
- **Issue:** Research narrative and script naming/output expectations can drift.
- **Upgrade:** autogenerate “current pipeline map” from existing scripts/configs.
- **Deliverable:** zero mismatch between docs and executable pipeline.

### P2.2 Add automated table builders for paper-ready summaries
- **Upgrade:**
  - Generate canonical CSV/LaTeX tables from JSON results:
    - index comparison table
    - routing table
    - eviction table
  - Include CI check that tables/figures regenerate from raw artifacts.
- **Deliverable:** no manual copy/paste errors in paper.

### P2.3 Add explicit threats-to-validity appendix artifact
- **Upgrade:** create structured validity notes from metadata:
  - external validity (dataset representativeness),
  - internal validity (measurement bias),
  - construct validity (proxy metric limits),
  - reproducibility constraints.
- **Deliverable:** stronger reviewer confidence and transparency.

---

## 5) Code-level change map (where to edit)

- **Benchmark integrity**
  - `src/benchmark/runner.py`
  - `src/benchmark/workload.py`
  - `scripts/03_run_index_benchmark.py`
  - `configs/index_benchmark.yaml`

- **Routing logic and evaluation rigor**
  - `src/router/adaptive_router.py`
  - `src/evaluation/routing_evaluator.py`
  - `src/router/cost_model.py` (verify consistency with measured cache latency)
  - `scripts/06_run_routing_eval.py`
  - `configs/routing.yaml`

- **Eviction upper-bound correctness and matrix completeness**
  - `src/cache/eviction/oracle.py`
  - `scripts/08_run_eviction.py`
  - `configs/eviction.yaml`

- **Reproducibility + tests**
  - `tests/` (new benchmark/routing/oracle-correctness tests)
  - `pyproject.toml` / environment docs for runtime pinning
  - all experiment scripts for standardized manifests

---

## 6) Journal-readiness acceptance checklist (completed workstreams)

- [ ] All benchmark workloads run with strict DB/query separation and validated cardinality.
- [ ] Final benchmark artifacts include workload dimension and complete manifests.
- [ ] “full” labels correspond to true full-scale data for the run.
- [ ] Oracle policy is verified as true upper bound under stream progression.
- [ ] Routing threshold optimization is constrained by robust quality criteria.
- [ ] Main routing/eviction results are reported on temporal splits.
- [ ] Eviction matrix is complete across all configured policies/cache sizes/seeds.
- [ ] Confidence intervals + significance tests accompany all headline claims.
- [ ] Tests guard scientific correctness assumptions (not only API behavior).
- [ ] Runtime/interpreter is pinned and reproducibly executable.
- [ ] README, scripts, outputs, and paper tables are fully consistent.

---

## 7) Recommended execution order (for fastest upgrade to paper quality)

1. **Benchmark integrity fixes first** (P0.1–P0.3), then regenerate benchmark artifacts.
2. **Oracle + routing logic fixes** (P0.4–P0.6), then regenerate routing/eviction artifacts.
3. **Complete eviction experiment matrix** (P0.7).
4. **Apply statistical and reproducibility upgrades** (P1.1–P1.5).
5. **Finalize manuscript consistency artifacts** (P2.1–P2.3).

This order maximizes scientific credibility early and avoids polishing results that may later be invalidated by core logic corrections.
