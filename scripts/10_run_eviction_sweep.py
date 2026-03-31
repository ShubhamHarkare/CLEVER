#!/usr/bin/env python3
"""
Phase 4b — Workload concentration sweep for eviction policies.

The key research experiment: systematically varies workload
concentration (gamma) and measures how each eviction policy
responds.  Produces the data for the phase diagram showing
when semantic-aware eviction helps vs. hurts.

gamma=0.0: uniform query distribution (no cluster structure)
gamma=0.5: moderate Zipf skew (realistic)
gamma=1.0: extreme concentration (nearly all queries from 1-2 topics)

Usage (local, dev data)::

    python scripts/10_run_eviction_sweep.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction_sweep.yaml \\
        --output results/eviction_sweep/

Usage (Great Lakes)::

    python scripts/10_run_eviction_sweep.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction_sweep.yaml \\
        --output results/eviction_sweep/ \\
        --multi-seed
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
_eviction_module = importlib.import_module("scripts.08_run_eviction")
create_policy = _eviction_module.create_policy
load_config = _eviction_module.load_config
load_data = _eviction_module.load_data
from src.cache.semantic_cache import SemanticCache
from src.cache.eviction.adaptive_semantic import AdaptiveSemanticPolicy
from src.benchmark.workload import (
    compute_workload_diversity,
    generate_concentrated_workload,
)
from src.utils.env_check import require_supported_runtime, pin_numpy_threads
from src.utils.manifest import generate_manifest

from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_policy_with_gamma(
    policy_name: str,
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    cache_size_pct: float,
    gamma: float,
    seed: int,
) -> dict:
    """Run eviction evaluation for one policy at a specific gamma.

    This is a modified version of evaluate_policy from 08_run_eviction
    that applies concentrated workload reordering via gamma.
    """
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    dim = embeddings.shape[1]

    # ── Data split (chronological) ───────────────────────────────
    warmup_pct = config["evaluation"].get("warmup_pct", 0.30)
    n_warmup = int(n * warmup_pct)
    max_cache_size = int(n * cache_size_pct)

    warmup_embs = embeddings[:n_warmup]
    warmup_texts = texts[:n_warmup]
    stream_embs = embeddings[n_warmup:]
    stream_texts = texts[n_warmup:]

    if n_warmup > max_cache_size:
        warmup_embs = warmup_embs[:max_cache_size]
        warmup_texts = warmup_texts[:max_cache_size]
        n_warmup = max_cache_size

    # ── Apply gamma-concentrated reordering to the stream ────────
    n_stream = len(stream_embs)
    if gamma > 0.001:
        reorder_indices = generate_concentrated_workload(
            query_vectors=stream_embs,
            n_queries=n_stream,
            gamma=gamma,
            seed=seed,
        )
        stream_embs = stream_embs[reorder_indices]
        stream_texts = [stream_texts[i] for i in reorder_indices]

    # ── Compute workload diversity of the reordered stream ───────
    diversity = compute_workload_diversity(stream_embs, seed=seed)

    logger.info(
        f"  [{policy_name}] gamma={gamma:.1f}, "
        f"diversity={diversity['cluster_entropy']:.3f}, "
        f"eff_clusters={diversity['effective_clusters']:.1f}, "
        f"cache={max_cache_size}, stream={n_stream}"
    )

    # ── Create policy ────────────────────────────────────────────
    cache_ids = list(range(n_warmup))
    policy = create_policy(
        policy_name, config, warmup_embs, cache_ids, stream_embs
    )

    # ── Build cache ──────────────────────────────────────────────
    cache_cfg = config.get("cache", {})
    t_build = time.perf_counter()
    cache = SemanticCache(
        dim=dim,
        index_type=cache_cfg.get("index_type", "hnsw"),
        index_params=cache_cfg.get("index_params", {}),
        max_size=max_cache_size,
        eviction_policy=policy,
    )
    cache.build(warmup_embs, warmup_texts)
    build_time = time.perf_counter() - t_build

    # ── Replay stream ────────────────────────────────────────────
    eval_cfg = config.get("evaluation", {})
    hit_threshold = eval_cfg.get("hit_threshold", 0.90)
    rolling_window = eval_cfg.get("rolling_window", 1000)
    log_interval = eval_cfg.get("log_interval", 500)

    n_hits = 0
    rolling_hits = deque(maxlen=rolling_window)
    checkpoints = []

    t_stream = time.perf_counter()

    for i in range(n_stream):
        query_emb = stream_embs[i]
        query_text = stream_texts[i]

        if policy_name == "oracle":
            policy.advance_stream(i)

        # Feed to adaptive policy for diversity tracking
        if hasattr(policy, "on_query"):
            policy.on_query(query_emb)

        result = cache.lookup(query_emb, k=1, threshold=hit_threshold)

        if result.hit:
            n_hits += 1
            rolling_hits.append(1)
        else:
            rolling_hits.append(0)
            cache.insert(query_emb, query_text)

        if (i + 1) % log_interval == 0 or i == n_stream - 1:
            cum_rate = n_hits / (i + 1)
            roll_rate = sum(rolling_hits) / len(rolling_hits) if rolling_hits else 0
            checkpoints.append({
                "query_idx": i + 1,
                "cumulative_hit_rate": round(cum_rate, 6),
                "rolling_hit_rate": round(roll_rate, 6),
                "n_evictions": cache._n_evictions,
                "cache_size": cache.size,
            })

    stream_time = time.perf_counter() - t_stream

    # ── Semantic coverage ────────────────────────────────────────
    sample_size = min(2000, n_stream)
    sample_idx = rng.choice(n_stream, sample_size, replace=False)
    dists, _ = cache.batch_lookup(stream_embs[sample_idx], k=1)
    coverage = float(np.mean(dists[:, 0]))

    # ── Collect adaptive alpha history if applicable ──────────────
    alpha_history = []
    if hasattr(policy, "_alpha_history"):
        alpha_history = policy._alpha_history

    final_hit_rate = n_hits / n_stream if n_stream > 0 else 0

    return {
        "policy": policy_name,
        "gamma": gamma,
        "cache_size_pct": cache_size_pct,
        "max_cache_size": max_cache_size,
        "seed": seed,
        "n_stream": n_stream,
        "final_hit_rate": round(final_hit_rate, 6),
        "n_hits": n_hits,
        "n_misses": n_stream - n_hits,
        "semantic_coverage_avg_dist": round(coverage, 6),
        "workload_diversity": diversity,
        "alpha_history": alpha_history,
        "cumulative_hit_rates": checkpoints,
        "timing": {
            "build_time_s": round(build_time, 3),
            "stream_time_s": round(stream_time, 3),
            "avg_query_time_ms": round(
                stream_time / n_stream * 1000, 4
            ) if n_stream > 0 else 0,
        },
        "cache_stats": cache.stats,
    }


def run_gamma_sweep(
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    seeds: list[int],
) -> dict:
    """Run the full gamma x policy x cache_size x seed sweep."""

    policies = config["eviction"]["policies"]
    cache_sizes = config["cache"]["cache_sizes_pct"]
    gamma_values = config.get("sweep", {}).get(
        "gamma_values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    total_runs = (
        len(gamma_values) * len(policies) * len(cache_sizes) * len(seeds)
    )
    logger.info(
        f"SWEEP: {len(gamma_values)} gammas × {len(policies)} policies × "
        f"{len(cache_sizes)} sizes × {len(seeds)} seeds = {total_runs} runs"
    )

    all_results = []
    run_num = 0

    for gamma in gamma_values:
        for policy_name in policies:
            for cache_pct in cache_sizes:
                for seed in seeds:
                    run_num += 1
                    logger.info(
                        f"RUN {run_num}/{total_runs}: "
                        f"γ={gamma:.1f} policy={policy_name} "
                        f"cache={cache_pct:.0%} seed={seed}"
                    )
                    try:
                        res = evaluate_policy_with_gamma(
                            policy_name, embeddings, texts, config,
                            cache_pct, gamma, seed,
                        )
                        all_results.append(res)
                    except Exception as e:
                        logger.error(f"FAILED: {e}", exc_info=True)
                        all_results.append({
                            "policy": policy_name,
                            "gamma": gamma,
                            "cache_size_pct": cache_pct,
                            "seed": seed,
                            "error": str(e),
                        })

    # ── Aggregation: group by (gamma, policy, cache_size) ────────
    aggregated = {}
    for gamma in gamma_values:
        g_key = f"{gamma:.2f}"
        aggregated[g_key] = {}
        for policy_name in policies:
            aggregated[g_key][policy_name] = {}
            for cache_pct in cache_sizes:
                p_key = f"{cache_pct:.2f}"
                matching = [
                    r for r in all_results
                    if r.get("gamma") == gamma
                    and r.get("policy") == policy_name
                    and r.get("cache_size_pct") == cache_pct
                    and "error" not in r
                ]
                if matching:
                    hrs = [r["final_hit_rate"] for r in matching]
                    covs = [r["semantic_coverage_avg_dist"] for r in matching]
                    divs = [
                        r["workload_diversity"]["cluster_entropy"]
                        for r in matching
                    ]
                    aggregated[g_key][policy_name][p_key] = {
                        "hit_rate_mean": round(float(np.mean(hrs)), 6),
                        "hit_rate_std": round(float(np.std(hrs)), 6),
                        "coverage_mean": round(float(np.mean(covs)), 6),
                        "coverage_std": round(float(np.std(covs)), 6),
                        "diversity_mean": round(float(np.mean(divs)), 4),
                        "n_seeds": len(matching),
                    }

    return {
        "manifest": generate_manifest(config),
        "sweep_results": all_results,
        "aggregated": aggregated,
        "config": {
            "policies": policies,
            "cache_sizes_pct": cache_sizes,
            "gamma_values": gamma_values,
            "seeds": seeds,
            "n_queries": len(embeddings),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4b: Eviction policy gamma sweep"
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument(
        "--config", default="configs/eviction_sweep.yaml",
    )
    parser.add_argument("--output", default="results/eviction_sweep/")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Override policies (e.g. --policies lru semantic adaptive)",
    )
    parser.add_argument(
        "--gammas", nargs="+", type=float, default=None,
        help="Override gamma values (e.g. --gammas 0.0 0.5 1.0)",
    )
    return parser.parse_args()


def main():
    require_supported_runtime()
    pin_numpy_threads()

    args = parse_args()
    config = load_config(args.config)

    if args.policies:
        config["eviction"]["policies"] = args.policies
    if args.gammas:
        config.setdefault("sweep", {})["gamma_values"] = args.gammas

    embeddings, texts = load_data(args.embeddings, args.queries)

    seeds = (
        config.get("seeds", [42, 123, 456])
        if args.multi_seed
        else [config.get("seed", 42)]
    )

    t_start = time.perf_counter()
    results = run_gamma_sweep(embeddings, texts, config, seeds)
    total_time = time.perf_counter() - t_start
    results["total_time_s"] = round(total_time, 1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sweep_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'═' * 60}")
    logger.info(f"SWEEP COMPLETE → {output_file}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"{'═' * 60}")

    # ── Summary table ────────────────────────────────────────────
    agg = results["aggregated"]
    policies = config["eviction"]["policies"]
    gamma_values = config.get("sweep", {}).get("gamma_values", [])
    cache_pct = config["cache"]["cache_sizes_pct"][0]  # show first size
    p_key = f"{cache_pct:.2f}"

    print(f"\n{'='*70}")
    print(f"HIT RATE BY GAMMA (cache={cache_pct:.0%})")
    print(f"{'='*70}")
    header = f"{'γ':<8}" + "".join(f"{p:<14}" for p in policies)
    print(header)
    print("-" * 70)
    for gamma in gamma_values:
        g_key = f"{gamma:.2f}"
        row = f"{gamma:<8.1f}"
        for p in policies:
            stats = agg.get(g_key, {}).get(p, {}).get(p_key, {})
            if stats:
                hr = stats["hit_rate_mean"]
                row += f"{hr:<14.4f}"
            else:
                row += f"{'—':<14}"
        print(row)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()