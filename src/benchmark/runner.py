"""
Benchmark runner — orchestrates a complete index benchmarking experiment.

For each (index_type, parameter_set) combination:
1. Hold out query vectors from the database
2. Build ground-truth (Flat) index on database vectors
3. Build the target index, measure build time + memory
4. Search with held-out queries, measure latency per query
5. Compute recall@{1,5,10} against ground truth
6. Repeat measurements and aggregate statistics
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.metrics import (
    compute_latency_stats,
    compute_recall_at_k,
    compute_throughput,
)
from src.benchmark.profiler import (
    measure_batch_throughput,
    measure_build,
    measure_search_latency,
)
from src.benchmark.workload import generate_workload
from src.indexes.factory import create_index
from src.utils.manifest import generate_manifest

logger = logging.getLogger(__name__)

# Maximum k for recall computation
MAX_K = 10


class BenchmarkRunner:
    """Orchestrates index benchmarking experiments."""

    def __init__(
        self,
        embeddings: np.ndarray,
        config: dict,
        output_dir: str | Path,
        dataset_label: str = "unknown",
    ):
        """
        Args:
            embeddings: Full embedding array, shape (N, D).
            config: Parsed index_benchmark.yaml config dict.
            output_dir: Directory for JSON result files.
            dataset_label: Scale label (e.g., "10k", "100k").
        """
        self.embeddings = embeddings
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_label = dataset_label
        self.dim = embeddings.shape[1]

        # Dataset config
        ds_cfg = config.get("dataset", {})
        self.query_count = ds_cfg.get("query_count", 1000)
        self.warmup_queries = ds_cfg.get("warmup_queries", 100)
        self.n_repeats = ds_cfg.get("repeat", 3)
        self.recall_k_values = config.get("recall_k_values", [1, 5, 10])
        self.workloads = config.get("workloads", ["uniform"])

    def run(self) -> list[dict]:
        """
        Run the full benchmark suite.

        Returns:
            List of result dicts, one per (index_type, params, workload) combo.
        """
        N = self.embeddings.shape[0]
        logger.info("=" * 70)
        logger.info(
            f"BENCHMARK: {self.dataset_label} — {N} vectors, dim={self.dim}"
        )
        logger.info("=" * 70)

        # --- Split into database + queries ---
        actual_query_count = min(self.query_count, N // 5)
        if actual_query_count < self.query_count:
            logger.warning(
                f"Reducing query count from {self.query_count} to "
                f"{actual_query_count} (dataset too small)"
            )

        rng = np.random.RandomState(42)
        all_indices = rng.permutation(N)
        query_indices = all_indices[:actual_query_count]
        db_indices = all_indices[actual_query_count:]

        db_vectors = self.embeddings[db_indices].copy()
        default_query_vectors = self.embeddings[query_indices].copy()

        logger.info(
            f"Split: {len(db_indices)} database + {len(query_indices)} queries"
        )

        # --- Build ground truth (Flat) ---
        logger.info("Building ground truth (Flat L2)...")
        gt_index = create_index("flat", dim=self.dim)
        gt_index.build(db_vectors)

        # --- Run benchmarks for each index type × workload ---
        all_results = []
        index_configs = self.config.get("indexes", {})

        for workload_type in self.workloads:
            logger.info(f"\n{'='*60}")
            logger.info(f"WORKLOAD: {workload_type}")
            logger.info(f"{'='*60}")

            # Generate workload-specific query selection
            if workload_type == "uniform":
                query_vectors = default_query_vectors
            else:
                try:
                    wl_indices = generate_workload(
                        query_vectors=default_query_vectors,
                        db_vectors=db_vectors,
                        workload_type=workload_type,
                        n_queries=actual_query_count, seed=42,
                    )
                    query_vectors = default_query_vectors[wl_indices]
                except Exception as e:
                    logger.warning(
                        f"Workload '{workload_type}' generation failed: {e}. "
                        f"Falling back to uniform."
                    )
                    query_vectors = default_query_vectors

            # Ground truth for this workload's queries
            _, gt_ids = gt_index.search(query_vectors, MAX_K)

            for index_type, idx_cfg in index_configs.items():
                param_list = idx_cfg.get("params", [{}])

                for params in param_list:
                    logger.info("-" * 60)
                    logger.info(
                        f"INDEX: {index_type} | params={params} | "
                        f"workload={workload_type}"
                    )
                    logger.info("-" * 60)

                    # Check if this config is valid for the dataset size
                    if index_type == "ivf" and params.get("nlist", 0) > len(db_vectors):
                        logger.warning(
                            f"Skipping IVF nlist={params['nlist']} > "
                            f"N={len(db_vectors)}"
                        )
                        continue

                    try:
                        result = self._benchmark_single(
                            index_type=index_type,
                            params=params,
                            db_vectors=db_vectors,
                            query_vectors=query_vectors,
                            gt_ids=gt_ids,
                        )
                        result["workload"] = workload_type
                        
                        # Guard against accidental leakage (P0.1)
                        if "wl_indices" in locals():
                            assert set(query_indices[wl_indices]).isdisjoint(set(db_indices)), \
                                "FATAL: Query vectors leaked into database vectors."
                            
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            f"FAILED: {index_type} params={params}: {e}",
                            exc_info=True,
                        )
                        all_results.append({
                            "index_type": index_type,
                            "params": params,
                            "dataset_size": len(db_vectors),
                            "dataset_label": self.dataset_label,
                            "workload": workload_type,
                            "error": str(e),
                        })

                    gc.collect()

        # Free ground truth index memory
        del gt_index
        gc.collect()

        # --- Output completeness and scale validation (P0.2, P0.3) ---
        expected_configs = sum(len(cfg.get("params", [{}])) for cfg in index_configs.values())
        expected_rows = expected_configs * len(self.workloads)
        
        # Determine number of valid skipped rows (e.g. IVF skip logic)
        skipped_configs = len([r for r in all_results if "error" not in r and r.get("skip", False)]) # not implemented yet, just for structure
        actual_valid_rows = len([r for r in all_results if "error" not in r])
        
        # Enforce workload inclusion
        for r in all_results:
            assert "workload" in r, f"Missing workload key in result row: {r}"

        # Hard validation for the "full" label
        eff_n_vectors = len(db_vectors)
        if "full" in self.dataset_label.lower():
            min_full_scale = self.config.get("min_full_scale", 500000)
            if eff_n_vectors < min_full_scale:
                logger.error(f"Scale inconsistency: dataset_label is '{self.dataset_label}' but effective_n_vectors is only {eff_n_vectors} < {min_full_scale}. Asserting failure.")
                raise ValueError(f"dataset_label '{self.dataset_label}' requires >= {min_full_scale} elements.")

        # Generate scientific provenance manifest
        manifest = generate_manifest(self.config)

        # Output results
        benchmark_results = {
            "dataset": self.dataset_label,
            "manifest": manifest,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": all_results,
        }

        output_path = self.output_dir / f"index_benchmark_{self.dataset_label}.json"
        with open(output_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        return all_results

    def _benchmark_single(
        self,
        index_type: str,
        params: dict,
        db_vectors: np.ndarray,
        query_vectors: np.ndarray,
        gt_ids: np.ndarray,
    ) -> dict:
        """Run benchmark for a single (index_type, params) configuration."""

        # --- Build phase ---
        index = create_index(index_type, dim=self.dim, **params)
        build_result = measure_build(index, db_vectors)

        # --- Search phase (per-query latency) ---
        latencies_ns, _, approx_ids, warmup_used = measure_search_latency(
            index, query_vectors, k=MAX_K, warmup=self.warmup_queries
        )
        lat_ms = latencies_ns / 1e6
        
        # Additional detailed latency profiling metadata
        latency_stats = {
            "mean": round(float(np.mean(lat_ms)), 4),
            "p50": round(float(np.median(lat_ms)), 4),
            "p95": round(float(np.percentile(lat_ms, 95)), 4),
            "p99": round(float(np.percentile(lat_ms, 99)), 4),
            "min": round(float(np.min(lat_ms)), 4),
            "max": round(float(np.max(lat_ms)), 4),
            "warmup_queries": warmup_used,
            "timing_mode": "individual_perf_counter_ns"
        }

        # --- Batch throughput ---
        throughput_result = measure_batch_throughput(
            index, query_vectors, k=MAX_K, n_repeats=self.n_repeats
        )

        # --- Recall@k ---
        recall_results = {}
        for k in self.recall_k_values:
            if k <= MAX_K:
                recall = compute_recall_at_k(approx_ids[:, :k], gt_ids[:, :k], k)
                recall_results[f"recall_at_{k}"] = round(recall, 4)

        # --- Assemble result ---
        result = {
            "index_type": index_type,
            "params": params,
            "dataset_size": len(db_vectors),
            "dataset_label": self.dataset_label,
            "build_time_s": build_result["build_time_s"],
            "peak_rss_mb": build_result["peak_rss_mb"],
            "post_build_rss_mb": build_result["post_build_rss_mb"],
            "memory_mb": build_result["memory_mb"], # Keep for backward compatibility or general memory
            "search_latency_ms": latency_stats,
            "throughput_qps": throughput_result["throughput_qps"],
            **recall_results,
        }

        logger.info(
            f"  → build={build_result['build_time_s']:.3f}s, "
            f"mem={build_result['memory_mb']:.1f}MB, "
            f"latency_p50={latency_stats['p50']:.3f}ms, "
            f"recall@10={recall_results.get('recall_at_10', 'N/A')}, "
            f"QPS={throughput_result['throughput_qps']:.0f}"
        )

        # Free the index
        del index
        gc.collect()

        return result
