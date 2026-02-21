#!/usr/bin/env python3
"""
Phase 3 — Cost-Based Query Routing Evaluation.

Loads embeddings, builds a semantic cache, and evaluates routing
strategies (fixed threshold, adaptive) with different index backends.

Usage:
    python scripts/06_run_routing_eval.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/routing.yaml \\
        --output results/routing/

    # With specific cache fill ratio:
    python scripts/06_run_routing_eval.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --fill-ratio 0.3 \\
        --output results/routing/
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.routing_evaluator import EvalConfig, RoutingEvaluator
from src.router.cost_model import CostModel
from src.utils.env_check import require_supported_runtime, pin_numpy_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(embeddings_path: str, queries_path: str) -> tuple:
    """Load embeddings and query texts."""
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path).astype(np.float32)
    logger.info(f"  Shape: {embeddings.shape}")

    logger.info(f"Loading queries from {queries_path}")
    df = pd.read_parquet(queries_path)
    texts = df["query_text"].tolist()
    logger.info(f"  Queries: {len(texts)}")

    # Build metadata from dataframe columns
    metadata = []
    for _, row in df.iterrows():
        meta = {}
        if "frequency" in df.columns:
            meta["frequency"] = int(row.get("frequency", 1))
        if "model" in df.columns:
            meta["model"] = row.get("model", "")
        if "language" in df.columns:
            meta["language"] = row.get("language", "")
        metadata.append(meta)

    if len(embeddings) != len(texts):
        logger.warning(
            f"Embedding count ({len(embeddings)}) != text count ({len(texts)}). "
            f"Truncating to min."
        )
        n = min(len(embeddings), len(texts))
        embeddings = embeddings[:n]
        texts = texts[:n]
        metadata = metadata[:n]

    return embeddings, texts, metadata


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_eval_config(config: dict) -> EvalConfig:
    """Build an EvalConfig from the YAML dict."""
    cache_cfg = config.get("cache", {})
    cost_cfg = config.get("cost_model", {})
    router_cfg = config.get("router", {})
    adaptive_cfg = router_cfg.get("adaptive", {})
    return EvalConfig(
        cache_fill_ratio=cache_cfg.get("fill_ratio", 0.5),
        index_type=cache_cfg.get("index_type", "hnsw"),
        index_params=cache_cfg.get("index_params", {}),
        thresholds=router_cfg.get("thresholds", []),
        min_quality=adaptive_cfg.get("min_quality", 0.8),
        calibration_ratio=adaptive_cfg.get("calibration_ratio", 0.2),
        llm_latency_ms=cost_cfg.get("llm_latency_ms", 500.0),
        llm_cost_usd=cost_cfg.get("llm_cost_usd", 0.01),
        fill_strategies=cache_cfg.get("fill_strategies", ["random"]),
        seed=config.get("seed", 42),
    )


def run_main_evaluation(
    embeddings: np.ndarray,
    texts: list[str],
    metadata: list[dict],
    config: dict,
    output_dir: Path,
):
    """Run the main routing evaluation (single seed)."""
    eval_config = _build_eval_config(config)

    evaluator = RoutingEvaluator(
        embeddings=embeddings,
        texts=texts,
        config=eval_config,
        metadata=metadata,
    )

    results = evaluator.run()
    evaluator.save(output_dir / "routing_eval.json")

    return results


def run_multi_seed_evaluation(
    embeddings: np.ndarray,
    texts: list[str],
    metadata: list[dict],
    config: dict,
    output_dir: Path,
):
    """Run routing evaluation across multiple seeds and aggregate."""
    eval_config = _build_eval_config(config)
    seeds = config.get("seeds", [42, 123, 456, 789, 1024])

    results = RoutingEvaluator.run_multi_seed(
        embeddings=embeddings,
        texts=texts,
        config=eval_config,
        seeds=seeds,
        metadata=metadata,
    )

    # Save aggregated results
    out_path = output_dir / "routing_eval_multi_seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Multi-seed results saved to {out_path}")

    return results


def run_index_comparison(
    embeddings: np.ndarray,
    texts: list[str],
    metadata: list[dict],
    config: dict,
    output_dir: Path,
):
    """Compare routing performance across different index backends."""
    comparison_configs = config.get("index_comparison", [])
    if not comparison_configs:
        logger.info("No index comparison configs found, skipping.")
        return {}

    cost_cfg = config.get("cost_model", {})
    router_cfg = config.get("router", {})
    adaptive_cfg = router_cfg.get("adaptive", {})

    comparison_results = {}

    for idx_cfg in comparison_configs:
        idx_type = idx_cfg["type"]
        idx_params = idx_cfg.get("params", {})
        label = f"{idx_type}_{_params_label(idx_params)}"
        logger.info(f"\n=== Index Comparison: {label} ===")

        eval_config = EvalConfig(
            cache_fill_ratio=config.get("cache", {}).get("fill_ratio", 0.5),
            index_type=idx_type,
            index_params=idx_params,
            thresholds=router_cfg.get("thresholds", []),
            min_quality=adaptive_cfg.get("min_quality", 0.8),
            calibration_ratio=adaptive_cfg.get("calibration_ratio", 0.2),
            llm_latency_ms=cost_cfg.get("llm_latency_ms", 500.0),
            llm_cost_usd=cost_cfg.get("llm_cost_usd", 0.01),
            fill_strategies=["random"],  # Only random for comparison
            seed=config.get("seed", 42),
        )

        try:
            evaluator = RoutingEvaluator(
                embeddings=embeddings,
                texts=texts,
                config=eval_config,
                metadata=metadata,
            )

            results = evaluator.run()
            comparison_results[label] = results
        except Exception as e:
            logger.error(f"Index {label} failed: {e}")
            comparison_results[label] = {"error": str(e)}

    # Save comparison
    out_path = output_dir / "index_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    logger.info(f"Index comparison saved to {out_path}")

    return comparison_results


def _params_label(params: dict) -> str:
    """Create a short label from params dict."""
    if not params:
        return "default"
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return "_".join(parts)


def print_summary(results: dict):
    """Print a summary of routing results."""
    logger.info("\n" + "=" * 70)
    logger.info("ROUTING EVALUATION SUMMARY")
    logger.info("=" * 70)

    for strategy, data in results.items():
        if strategy == "meta":
            continue
        logger.info(f"\n--- Strategy: {strategy} ---")
        logger.info(f"  Cache size:  {data['cache_size']}")
        logger.info(f"  Eval size:   {data['eval_size']}")
        logger.info(f"  Build time:  {data['build_time_s']:.3f}s")
        logger.info(f"  Memory:      {data['cache_memory_mb']:.1f}MB")

        # Best threshold from sweep
        sweep = data.get("threshold_sweep", [])
        if sweep:
            best = max(sweep, key=lambda x: x["latency_saving_pct"])
            logger.info(f"  Best fixed threshold:")
            logger.info(f"    Threshold:      {best['threshold']}")
            logger.info(f"    Hit rate:       {best['hit_rate']:.2%}")
            logger.info(f"    Cosine quality: {best.get('cosine_quality', 'N/A')}")
            logger.info(f"    Recall@1:       {best.get('recall_at_1', 'N/A')}")
            logger.info(f"    Latency saving: {best['latency_saving_pct']:.1f}%")
            logger.info(f"    Cost saving:    {best['monetary_saving_pct']:.1f}%")

        # Adaptive result
        adaptive = data.get("adaptive", {})
        if adaptive:
            logger.info(f"  Adaptive router:")
            logger.info(f"    Threshold:      {adaptive['best_threshold']}")
            logger.info(f"    Hit rate:       {adaptive['test_hit_rate']:.2%}")
            logger.info(f"    Cosine quality: {adaptive.get('cosine_quality', 'N/A')}")
            logger.info(f"    Recall@1:       {adaptive.get('recall_at_1', 'N/A')}")
            logger.info(f"    Latency saving: {adaptive['latency_saving_pct']:.1f}%")


def print_multi_seed_summary(results: dict):
    """Print aggregated multi-seed results."""
    logger.info("\n" + "=" * 70)
    logger.info(f"MULTI-SEED SUMMARY  ({results['n_seeds']} seeds: {results['seeds']})")
    logger.info("=" * 70)

    for strategy, metrics in results.get("aggregated", {}).items():
        logger.info(f"\n--- Strategy: {strategy} ---")
        for key, stats in metrics.items():
            logger.info(
                f"  {key:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )


def main():
    require_supported_runtime()
    pin_numpy_threads()
    
    parser = argparse.ArgumentParser(
        description="Phase 3 — Cost-Based Query Routing Evaluation"
    )
    parser.add_argument(
        "--embeddings", type=str,
        default="results/embeddings/full_embeddings.npy",
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--queries", type=str,
        default="data/full_queries.parquet",
        help="Path to queries .parquet file",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/routing.yaml",
        help="Path to routing config YAML",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/routing/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--fill-ratio", type=float, default=None,
        help="Override cache fill ratio from config",
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip index comparison evaluation",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run evaluation across multiple seeds and aggregate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    embeddings, texts, metadata = load_data(args.embeddings, args.queries)

    # Load config
    config = load_config(args.config)

    # Override fill ratio if specified
    if args.fill_ratio is not None:
        config.setdefault("cache", {})["fill_ratio"] = args.fill_ratio

    if args.multi_seed:
        # Multi-seed evaluation
        results = run_multi_seed_evaluation(
            embeddings, texts, metadata, config, output_dir
        )
        print_multi_seed_summary(results)
    else:
        # Single-seed evaluation (default)
        results = run_main_evaluation(
            embeddings, texts, metadata, config, output_dir
        )
        print_summary(results)

    # Run index comparison
    if not args.skip_comparison:
        run_index_comparison(
            embeddings, texts, metadata, config, output_dir
        )

    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()

