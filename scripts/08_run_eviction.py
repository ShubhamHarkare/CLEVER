#!/usr/bin/env python3
"""
Phase 4 — Eviction policy evaluation.

Simulates a realistic cache lifecycle:
1. **Warmup**: Build a bounded-size cache from the first portion of queries.
2. **Stream**: Replay the remaining queries as a live query stream.
   - On **hit**: record success, update eviction bookkeeping.
   - On **miss**: insert the new query into the cache (may trigger eviction).
3. **Metrics**: Track cumulative hit rate, rolling hit rate, semantic
   coverage, and eviction overhead per policy.

Usage
-----
Local smoke test (8.5K dev subset)::

    python scripts/08_run_eviction.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction.yaml \\
        --output results/eviction/

Great Lakes (full dataset)::

    python scripts/08_run_eviction.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction.yaml \\
        --output results/eviction/ \\
        --multi-seed
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache.semantic_cache import SemanticCache
from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy
from src.cache.eviction.oracle import OraclePolicy
from src.utils.env_check import require_supported_runtime, pin_numpy_threads
from src.utils.manifest import generate_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(
    embeddings_path: str, queries_path: str
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and query texts."""
    embeddings = np.load(embeddings_path).astype(np.float32)
    df = pd.read_parquet(queries_path)
    texts = df["query_text"].tolist()

    # Truncate to match (in case lengths differ)
    n = min(len(embeddings), len(texts))
    embeddings = embeddings[:n]
    texts = texts[:n]

    logger.info(f"Loaded {n} queries, embedding dim={embeddings.shape[1]}")
    return embeddings, texts


def create_policy(
    policy_name: str,
    config: dict,
    cache_embs: np.ndarray,
    cache_ids: list[int],
    stream_embs: np.ndarray,
) -> "EvictionPolicy":
    """Create an eviction policy instance from config.

    Args:
        policy_name: One of 'lru', 'lfu', 'semantic', 'oracle'.
        config: Full experiment config dict.
        cache_embs: Initial cache embeddings (for oracle pre-computation).
        cache_ids: Initial cache IDs.
        stream_embs: Future query stream embeddings (for oracle).
    """
    eviction_cfg = config.get("eviction", {})

    if policy_name == "lru":
        return LRUPolicy()
    elif policy_name == "lfu":
        return LFUPolicy()
    elif policy_name == "semantic":
        sem_cfg = eviction_cfg.get("semantic", {})
        return SemanticPolicy(
            similarity_threshold=sem_cfg.get("similarity_threshold", 0.30),
            alpha=sem_cfg.get("alpha", 1.0),
            beta=sem_cfg.get("beta", 1.0),
            recompute_interval=sem_cfg.get("recompute_interval", 50),
        )
    elif policy_name == "oracle":
        return OraclePolicy(
            future_stream_embeddings=stream_embs,
            cache_embeddings=cache_embs,
            cache_ids=cache_ids,
            similarity_threshold=config["evaluation"].get("hit_threshold", 0.90),
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


# ═════════════════════════════════════════════════════════════════
# Core evaluation
# ═════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_name: str,
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    cache_size_pct: float,
    seed: int,
) -> dict:
    """Run a single eviction evaluation for one policy + cache size.

    Returns:
        Dict with cumulative_hit_rate, rolling_hit_rates, semantic
        coverage, eviction stats, and per-phase timing.
    """
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    dim = embeddings.shape[1]

    # ── Data split ───────────────────────────────────────────────
    warmup_pct = config["evaluation"].get("warmup_pct", 0.30)
    n_warmup = int(n * warmup_pct)
    max_cache_size = int(n * cache_size_pct)

    # (P0.6) Do NOT shuffle data — enforce chronological/temporal splitting
    # for realistic evaluation of concept drift and caching
    shuffled_embs = embeddings
    shuffled_texts = texts

    # Split: warmup entries fill the cache, rest is the stream
    warmup_embs = shuffled_embs[:n_warmup]
    warmup_texts = shuffled_texts[:n_warmup]
    stream_embs = shuffled_embs[n_warmup:]
    stream_texts = shuffled_texts[n_warmup:]

    # If warmup is larger than max_cache_size, truncate warmup to fill cache
    if n_warmup > max_cache_size:
        warmup_embs = warmup_embs[:max_cache_size]
        warmup_texts = warmup_texts[:max_cache_size]
        n_warmup = max_cache_size

    n_stream = len(stream_embs)

    logger.info(
        f"  [{policy_name}] cache_size={max_cache_size} "
        f"({cache_size_pct:.0%}), warmup={n_warmup}, stream={n_stream}, "
        f"seed={seed}"
    )

    # ── Create policy ────────────────────────────────────────────
    cache_ids = list(range(n_warmup))
    policy = create_policy(
        policy_name, config, warmup_embs, cache_ids, stream_embs
    )

    # ── Build cache ──────────────────────────────────────────────
    cache_cfg = config.get("cache", {})
    t_build_start = time.perf_counter()
    cache = SemanticCache(
        dim=dim,
        index_type=cache_cfg.get("index_type", "hnsw"),
        index_params=cache_cfg.get("index_params", {}),
        max_size=max_cache_size,
        eviction_policy=policy,
    )
    cache.build(warmup_embs, warmup_texts)
    build_time = time.perf_counter() - t_build_start

    # ── Replay query stream ──────────────────────────────────────
    eval_cfg = config.get("evaluation", {})
    hit_threshold = eval_cfg.get("hit_threshold", 0.90)
    rolling_window = eval_cfg.get("rolling_window", 1000)
    log_interval = eval_cfg.get("log_interval", 5000)

    n_hits = 0
    rolling_hits = deque(maxlen=rolling_window)
    cumulative_hit_rates = []
    rolling_hit_rates = []
    eviction_timestamps = []  # (stream_idx, n_evictions_so_far)

    t_stream_start = time.perf_counter()

    for i in range(n_stream):
        query_emb = stream_embs[i]
        query_text = stream_texts[i]

        # Advance oracle stream position
        if policy_name == "oracle":
            policy.advance_stream(i)

        # Lookup
        result = cache.lookup(query_emb, k=1, threshold=hit_threshold)

        if result.hit:
            n_hits += 1
            rolling_hits.append(1)
        else:
            rolling_hits.append(0)
            # Insert on miss → may trigger eviction
            cache.insert(query_emb, query_text)

        # Record metrics periodically
        if (i + 1) % log_interval == 0 or i == n_stream - 1:
            cum_rate = n_hits / (i + 1)
            roll_rate = sum(rolling_hits) / len(rolling_hits) if rolling_hits else 0
            cumulative_hit_rates.append({
                "query_idx": i + 1,
                "cumulative_hit_rate": round(cum_rate, 6),
                "rolling_hit_rate": round(roll_rate, 6),
                "n_evictions": cache._n_evictions,
                "cache_size": cache.size,
            })

            if (i + 1) % log_interval == 0:
                logger.info(
                    f"    [{policy_name}] {i+1}/{n_stream} queries — "
                    f"cum_hit={cum_rate:.4f} roll_hit={roll_rate:.4f} "
                    f"evictions={cache._n_evictions} size={cache.size}"
                )

    stream_time = time.perf_counter() - t_stream_start

    # ── Compute semantic coverage ────────────────────────────────
    # Avg min-distance from stream queries to their nearest cached entry
    # Lower = better coverage (the cache covers more of the query space)
    t_coverage_start = time.perf_counter()
    sample_size = min(2000, n_stream)
    sample_idx = rng.choice(n_stream, sample_size, replace=False)
    sample_embs = stream_embs[sample_idx]
    dists, _ = cache.batch_lookup(sample_embs, k=1)
    semantic_coverage = float(np.mean(dists[:, 0]))
    coverage_time = time.perf_counter() - t_coverage_start

    # ── Collect results ──────────────────────────────────────────
    final_hit_rate = n_hits / n_stream if n_stream > 0 else 0
    cache_stats = cache.stats

    result = {
        "policy": policy_name,
        "cache_size_pct": cache_size_pct,
        "max_cache_size": max_cache_size,
        "seed": seed,
        "n_queries": n,
        "n_warmup": n_warmup,
        "n_stream": n_stream,
        "final_hit_rate": round(final_hit_rate, 6),
        "n_hits": n_hits,
        "n_misses": n_stream - n_hits,
        "semantic_coverage_avg_dist": round(semantic_coverage, 6),
        "cumulative_hit_rates": cumulative_hit_rates,
        "timing": {
            "build_time_s": round(build_time, 3),
            "stream_time_s": round(stream_time, 3),
            "coverage_time_s": round(coverage_time, 3),
            "avg_query_time_ms": round(
                stream_time / n_stream * 1000, 4
            ) if n_stream > 0 else 0,
        },
        "cache_stats": cache_stats,
    }

    logger.info(
        f"  [{policy_name}] DONE — hit_rate={final_hit_rate:.4f} "
        f"evictions={cache_stats['n_evictions']} "
        f"coverage={semantic_coverage:.4f} "
        f"time={stream_time:.1f}s"
    )

    return result


# ═════════════════════════════════════════════════════════════════
# Multi-seed runner
# ═════════════════════════════════════════════════════════════════

def run_full_experiment(
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    seeds: list[int],
    max_workers: int = 1,
) -> dict:
    """Run all policy × cache_size × seed combinations in parallel.

    Returns:
        Dict with nested results: results[policy][cache_size_pct][seed]
        plus aggregated statistics.
    """
    policies = config["eviction"]["policies"]
    cache_sizes = config["cache"]["cache_sizes_pct"]

    all_results = {}
    aggregated = {}

    total_runs = len(policies) * len(cache_sizes) * len(seeds)
    run_num = 0

    # Prepare data structure
    for policy_name in policies:
        all_results[policy_name] = {}
        aggregated[policy_name] = {}
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            all_results[policy_name][pct_key] = []
            
    tasks = []
    
    # If only 1 worker, run sequentially to avoid ProcessPool overhead and simplify logging
    if max_workers <= 1:
        for policy_name in policies:
            for cache_pct in cache_sizes:
                for seed in seeds:
                    run_num += 1
                    logger.info(
                        f"\n{'═' * 60}\n"
                        f"RUN {run_num}/{total_runs}: "
                        f"policy={policy_name}, cache={cache_pct:.0%}, seed={seed}\n"
                        f"{'═' * 60}"
                    )
                    res = evaluate_policy(
                        policy_name=policy_name,
                        embeddings=embeddings,
                        texts=texts,
                        config=config,
                        cache_size_pct=cache_pct,
                        seed=seed
                    )
                    all_results[policy_name][f"{cache_pct:.2f}"].append(res)
    else:
        logger.info(f"Starting parallel execution with {max_workers} workers for {total_runs} runs.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {}
            for policy_name in policies:
                for cache_pct in cache_sizes:
                    for seed in seeds:
                        future = executor.submit(
                            evaluate_policy,
                            policy_name, embeddings, texts, config, cache_pct, seed
                        )
                        future_to_info[future] = (policy_name, f"{cache_pct:.2f}")

            for future in as_completed(future_to_info):
                policy_name, pct_key = future_to_info[future]
                try:
                    res = future.result()
                    all_results[policy_name][pct_key].append(res)
                    run_num += 1
                    logger.info(f"Completed {run_num}/{total_runs} (Policy: {policy_name}, Cache: {pct_key}, Seed: {res['seed']})")
                except Exception as exc:
                    logger.error(f"Task generated an exception: {exc}")

    # Aggregation Phase
    for policy_name in policies:
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            seed_results = all_results[policy_name][pct_key]
            
            # Sort back by seed to ensure deterministic output array ordering
            seed_results.sort(key=lambda x: x["seed"])

            hit_rates = [r["final_hit_rate"] for r in seed_results]
            coverages = [r["semantic_coverage_avg_dist"] for r in seed_results]
            stream_times = [r["timing"]["stream_time_s"] for r in seed_results]
            eviction_counts = [r["cache_stats"]["n_evictions"] for r in seed_results]

            aggregated[policy_name][pct_key] = {
                "hit_rate_mean": round(float(np.mean(hit_rates)), 6),
                "hit_rate_std": round(float(np.std(hit_rates)), 6),
                "coverage_mean": round(float(np.mean(coverages)), 6),
                "coverage_std": round(float(np.std(coverages)), 6),
                "stream_time_mean_s": round(float(np.mean(stream_times)), 3),
                "evictions_mean": round(float(np.mean(eviction_counts)), 1),
                "n_seeds": len(seeds),
                "seeds": seeds,
            }

    return {
        "manifest": generate_manifest(config),
        "per_seed": all_results,
        "aggregated": aggregated,
        "config": {
            "policies": policies,
            "cache_sizes_pct": cache_sizes,
            "seeds": seeds,
            "n_queries": len(embeddings),
        },
    }


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Eviction policy evaluation"
    )
    parser.add_argument(
        "--embeddings", required=True,
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--queries", required=True,
        help="Path to queries .parquet file",
    )
    parser.add_argument(
        "--config", default="configs/eviction.yaml",
        help="Path to eviction config YAML",
    )
    parser.add_argument(
        "--output", default="results/eviction/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run with multiple seeds from config",
    )
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Override policies to run (e.g. --policies lru semantic)",
    )
    parser.add_argument(
        "--cache-sizes", nargs="+", type=float, default=None,
        help="Override cache sizes (e.g. --cache-sizes 0.10 0.20)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for multi-processing. Defaults to 1 (sequential).",
    )
    return parser.parse_args()


def main():
    require_supported_runtime()
    pin_numpy_threads()
    
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.policies:
        config["eviction"]["policies"] = args.policies
    if args.cache_sizes:
        config["cache"]["cache_sizes_pct"] = args.cache_sizes

    # Load data
    embeddings, texts = load_data(args.embeddings, args.queries)

    # Determine seeds
    if args.multi_seed:
        seeds = config.get("seeds", [42, 123, 456])
    else:
        seeds = [config.get("seed", 42)]

    # Run experiment
    t_start = time.perf_counter()
    results = run_full_experiment(embeddings, texts, config, seeds, max_workers=args.workers)
    total_time = time.perf_counter() - t_start

    results["total_time_s"] = round(total_time, 1)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_multi_seed" if args.multi_seed else ""
    output_file = output_dir / f"eviction_results{suffix}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'═' * 60}")
    logger.info(f"RESULTS SAVED → {output_file}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"{'═' * 60}")

    # Print summary table
    print("\n\n" + "=" * 70)
    print("EVICTION POLICY COMPARISON")
    print("=" * 70)
    agg = results["aggregated"]
    print(f"{'Policy':<12} {'Cache %':<10} {'Hit Rate':<18} {'Coverage':<18} {'Evictions':<12}")
    print("-" * 70)
    for policy in config["eviction"]["policies"]:
        for pct in config["cache"]["cache_sizes_pct"]:
            pct_key = f"{pct:.2f}"
            if policy in agg and pct_key in agg[policy]:
                a = agg[policy][pct_key]
                hr = f"{a['hit_rate_mean']:.4f} ± {a['hit_rate_std']:.4f}"
                cov = f"{a['coverage_mean']:.4f} ± {a['coverage_std']:.4f}"
                ev = f"{a['evictions_mean']:.0f}"
                print(f"{policy:<12} {pct:<10.0%} {hr:<18} {cov:<18} {ev:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
