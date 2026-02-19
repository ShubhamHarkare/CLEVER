"""
Routing evaluation pipeline.

Simulates a realistic query stream scenario:
1. Seed the cache with a subset of queries (cache warmup)
2. Route remaining queries through the router
3. Measure cost savings, hit rates, and quality

Supports multiple cache fill strategies and evaluates both
fixed-threshold and adaptive routing.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.cache.semantic_cache import SemanticCache
from src.indexes.factory import create_index
from src.router.adaptive_router import AdaptiveRouter
from src.router.cost_model import CostModel
from src.router.similarity_router import SimilarityRouter, RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for routing evaluation."""

    # Cache configuration
    cache_fill_ratio: float = 0.5  # Fraction of data used to fill cache
    index_type: str = "hnsw"
    index_params: dict = field(default_factory=lambda: {
        "M": 32, "efConstruction": 128, "efSearch": 128,
    })

    # Router configuration
    thresholds: list[float] = field(
        default_factory=lambda: [
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
        ]
    )

    # Adaptive router
    min_quality: float = 0.8
    calibration_ratio: float = 0.2  # fraction of eval set for calibration

    # Cost model
    llm_latency_ms: float = 500.0
    llm_cost_usd: float = 0.01

    # Fill strategies to evaluate
    fill_strategies: list[str] = field(
        default_factory=lambda: ["random", "frequency"]
    )

    # Reproducibility
    seed: int = 42


class RoutingEvaluator:
    """Evaluates routing strategies on a query dataset.

    Workflow:
    1. Split data → cache pool + evaluation queries
    2. Build the semantic cache from the cache pool
    3. For each threshold, route eval queries and measure performance
    4. Run adaptive routing to find optimal threshold
    5. Compare routing across different index backends
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        config: EvalConfig,
        metadata: Optional[list[dict]] = None,
    ):
        """
        Args:
            embeddings: All embeddings, shape (N, D).
            texts: All query texts.
            config: Evaluation configuration.
            metadata: Optional per-query metadata.
        """
        self.embeddings = embeddings
        self.texts = texts
        self.config = config
        self.metadata = metadata or [{}] * len(texts)
        self.dim = embeddings.shape[1]
        self.results: dict = {}

    def run(self) -> dict:
        """Run the full evaluation pipeline.

        Returns:
            Dict with all results including threshold sweep,
            adaptive routing, and index comparison.
        """
        t_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info(
            f"ROUTING EVALUATION: {len(self.embeddings)} queries, "
            f"dim={self.dim}"
        )
        logger.info("=" * 70)

        all_results = {}

        for strategy in self.config.fill_strategies:
            logger.info(f"\n--- Fill strategy: {strategy} ---")
            result = self._evaluate_strategy(strategy)
            all_results[strategy] = result

        total_time = time.perf_counter() - t_start
        all_results["meta"] = {
            "total_queries": len(self.embeddings),
            "dim": self.dim,
            "total_time_s": round(total_time, 2),
            "seed": self.config.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": {
                "cache_fill_ratio": self.config.cache_fill_ratio,
                "index_type": self.config.index_type,
                "index_params": self.config.index_params,
                "llm_latency_ms": self.config.llm_latency_ms,
                "llm_cost_usd": self.config.llm_cost_usd,
            },
        }

        self.results = all_results
        return all_results

    def _evaluate_strategy(self, strategy: str) -> dict:
        """Evaluate a single cache fill strategy."""

        # --- Split data ---
        cache_emb, cache_texts, eval_emb, eval_texts = self._split_data(
            strategy
        )
        logger.info(
            f"Split: {len(cache_emb)} cache + {len(eval_emb)} eval queries"
        )

        # --- Build semantic cache ---
        cache = SemanticCache(
            dim=self.dim,
            index_type=self.config.index_type,
            index_params=self.config.index_params,
        )
        build_time = cache.build(cache_emb, cache_texts)
        logger.info(f"Cache built in {build_time:.3f}s ({cache.size} entries)")

        # --- Build ground truth index on cache entries ---
        gt_index = create_index("flat", dim=self.dim)
        gt_index.build(cache_emb.astype(np.float32))

        # --- Batch search: eval queries against cache ---
        cache_distances, cache_indices = cache.batch_lookup(eval_emb, k=10)

        # --- Ground truth: eval queries against cache (exact) ---
        gt_distances, gt_indices = gt_index.search(
            eval_emb.astype(np.float32), k=10
        )

        # --- Cost model ---
        cost_model = CostModel(
            llm_latency_ms=self.config.llm_latency_ms,
            llm_cost_usd=self.config.llm_cost_usd,
            cache_latency_ms=0.5,  # Will be updated from actual lookup
            index_type=self.config.index_type,
            index_params=self.config.index_params,
            cache_memory_mb=cache.memory_mb,
            cache_build_time_s=build_time,
        )

        # --- Threshold sweep ---
        threshold_results = self._threshold_sweep(
            cache_distances, gt_distances, cache_indices, gt_indices,
            cost_model,
        )

        # --- Adaptive routing ---
        adaptive_result = self._adaptive_routing(
            cache_distances, gt_distances, cache_indices, gt_indices,
            cost_model,
        )

        # --- Per-query distance analysis ---
        distance_analysis = self._distance_analysis(
            cache_distances, eval_texts
        )

        return {
            "cache_size": cache.size,
            "eval_size": len(eval_emb),
            "build_time_s": round(build_time, 3),
            "cache_memory_mb": round(cache.memory_mb, 2),
            "cost_model": cost_model.summary(),
            "threshold_sweep": threshold_results,
            "adaptive": adaptive_result,
            "distance_analysis": distance_analysis,
            "cache_stats": cache.stats,
        }

    def _split_data(
        self, strategy: str
    ) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
        """Split data into cache and evaluation sets.

        Args:
            strategy: "random", "frequency", or "chronological".

        Returns:
            (cache_emb, cache_texts, eval_emb, eval_texts)
        """
        n = len(self.embeddings)
        n_cache = int(n * self.config.cache_fill_ratio)

        if strategy == "frequency":
            # Higher-frequency queries go to cache (more likely to be reused)
            freqs = np.array([
                m.get("frequency", 1) for m in self.metadata
            ], dtype=float)
            # Sort by frequency descending; top n_cache go to cache
            sorted_idx = np.argsort(-freqs)
            cache_idx = sorted_idx[:n_cache]
            eval_idx = sorted_idx[n_cache:]
        elif strategy == "chronological":
            # First n_cache queries fill cache, rest are evaluation
            cache_idx = np.arange(n_cache)
            eval_idx = np.arange(n_cache, n)
        else:
            # Random split
            rng = np.random.RandomState(self.config.seed)
            perm = rng.permutation(n)
            cache_idx = perm[:n_cache]
            eval_idx = perm[n_cache:]

        cache_emb = self.embeddings[cache_idx]
        cache_texts = [self.texts[i] for i in cache_idx]
        eval_emb = self.embeddings[eval_idx]
        eval_texts = [self.texts[i] for i in eval_idx]

        return cache_emb, cache_texts, eval_emb, eval_texts

    @staticmethod
    def _compute_quality(
        cache_nn_dist: np.ndarray,
        gt_nn_dist: np.ndarray,
        cache_nn_ids: np.ndarray,
        gt_nn_ids: np.ndarray,
        is_hit: np.ndarray,
    ) -> dict:
        """Compute multi-faceted quality metrics for cache hits.

        Returns dict with:
            cosine_quality:  mean exp(-α * dist) over hits; decays with distance.
            recall_at_1:     fraction of hits where cache NN == ground-truth NN.
            distance_quality: legacy binary proxy (dist <= 2 * gt + 0.01).
        """
        n_hits = int(is_hit.sum())
        if n_hits == 0:
            return {"cosine_quality": 1.0, "recall_at_1": 1.0, "distance_quality": 1.0}

        hit_dists = cache_nn_dist[is_hit]
        hit_gt_dists = gt_nn_dist[is_hit]

        # 1) Cosine-similarity-based quality (exponential decay)
        #    For normalized vectors: cosine_sim = 1 - L2²/2
        #    quality = mean(cosine_sim) mapped to [0, 1]
        cosine_sims = np.clip(1.0 - hit_dists / 2.0, 0.0, 1.0)
        cosine_quality = float(np.mean(cosine_sims))

        # 2) Recall@1: does the cache return the same NN as ground truth?
        hit_cache_ids = cache_nn_ids[is_hit]
        hit_gt_ids = gt_nn_ids[is_hit]
        recall_at_1 = float(np.mean(hit_cache_ids == hit_gt_ids))

        # 3) Legacy distance proxy (backward compat)
        distance_quality = float(np.mean(hit_dists <= hit_gt_dists * 2.0 + 0.01))

        return {
            "cosine_quality": round(cosine_quality, 4),
            "recall_at_1": round(recall_at_1, 4),
            "distance_quality": round(distance_quality, 4),
        }

    def _threshold_sweep(
        self,
        cache_distances: np.ndarray,
        gt_distances: np.ndarray,
        cache_indices: np.ndarray,
        gt_indices: np.ndarray,
        cost_model: CostModel,
    ) -> list[dict]:
        """Sweep thresholds and measure performance at each."""
        nn_dist = cache_distances[:, 0]
        gt_nn_dist = gt_distances[:, 0]
        cache_nn_ids = cache_indices[:, 0]
        gt_nn_ids = gt_indices[:, 0]
        n = len(nn_dist)
        results = []

        for threshold in self.config.thresholds:
            is_hit = nn_dist <= threshold
            n_hits = int(is_hit.sum())
            hit_rate = n_hits / n if n > 0 else 0

            quality = self._compute_quality(
                nn_dist, gt_nn_dist, cache_nn_ids, gt_nn_ids, is_hit,
            )

            savings = cost_model.cost_savings(hit_rate)

            results.append({
                "threshold": round(float(threshold), 4),
                "hit_rate": round(hit_rate, 4),
                **quality,
                "n_hits": n_hits,
                "n_misses": n - n_hits,
                **savings,
            })

        return results

    def _adaptive_routing(
        self,
        cache_distances: np.ndarray,
        gt_distances: np.ndarray,
        cache_indices: np.ndarray,
        gt_indices: np.ndarray,
        cost_model: CostModel,
    ) -> dict:
        """Run adaptive router calibration."""
        # Split eval data into calibration + test
        n = len(cache_distances)
        n_cal = max(10, int(n * self.config.calibration_ratio))

        rng = np.random.RandomState(self.config.seed + 1)
        perm = rng.permutation(n)
        cal_idx = perm[:n_cal]
        test_idx = perm[n_cal:]

        # Calibrate on calibration set
        adaptive = AdaptiveRouter(
            cost_model=cost_model,
            min_quality=self.config.min_quality,
        )
        best_threshold = adaptive.calibrate(
            cache_distances[cal_idx],
            gt_distances[cal_idx],
        )

        # Evaluate on test set
        test_nn_dist = cache_distances[test_idx, 0]
        test_gt_dist = gt_distances[test_idx, 0]
        test_cache_ids = cache_indices[test_idx, 0]
        test_gt_ids = gt_indices[test_idx, 0]
        is_hit = test_nn_dist <= best_threshold
        n_hits = int(is_hit.sum())
        n_test = len(test_idx)
        hit_rate = n_hits / n_test if n_test > 0 else 0

        quality = self._compute_quality(
            test_nn_dist, test_gt_dist, test_cache_ids, test_gt_ids, is_hit,
        )

        savings = cost_model.cost_savings(hit_rate)

        return {
            "best_threshold": round(best_threshold, 4),
            "calibration_size": n_cal,
            "test_size": n_test,
            "test_hit_rate": round(hit_rate, 4),
            **quality,
            **savings,
            "sweep": adaptive.sweep_summary(),
        }

    def _distance_analysis(
        self,
        cache_distances: np.ndarray,
        eval_texts: list[str],
    ) -> dict:
        """Analyze the distribution of cache distances."""
        nn_dist = cache_distances[:, 0]

        return {
            "mean": round(float(nn_dist.mean()), 4),
            "std": round(float(nn_dist.std()), 4),
            "median": round(float(np.median(nn_dist)), 4),
            "min": round(float(nn_dist.min()), 4),
            "max": round(float(nn_dist.max()), 4),
            "p5": round(float(np.percentile(nn_dist, 5)), 4),
            "p25": round(float(np.percentile(nn_dist, 25)), 4),
            "p75": round(float(np.percentile(nn_dist, 75)), 4),
            "p95": round(float(np.percentile(nn_dist, 95)), 4),
            "histogram": {
                "bins": [round(b, 2) for b in np.linspace(0, 2, 21).tolist()],
                "counts": np.histogram(nn_dist, bins=np.linspace(0, 2, 21))[0].tolist(),
            },
        }

    def save(self, output_path: str | Path):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    # ------------------------------------------------------------------
    # Multi-seed aggregation
    # ------------------------------------------------------------------

    @classmethod
    def run_multi_seed(
        cls,
        embeddings: np.ndarray,
        texts: list[str],
        config: EvalConfig,
        seeds: list[int],
        metadata: Optional[list[dict]] = None,
    ) -> dict:
        """Run the evaluation pipeline with multiple seeds and aggregate.

        Args:
            embeddings: All embeddings, shape (N, D).
            texts: All query texts.
            config: Base evaluation configuration.
            seeds: List of random seeds to evaluate.
            metadata: Optional per-query metadata.

        Returns:
            Dict with ``per_seed`` results and ``aggregated`` summary
            (mean ± std for key metrics).
        """
        per_seed_results = {}

        for i, seed in enumerate(seeds):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"MULTI-SEED RUN {i+1}/{len(seeds)}  (seed={seed})")
            logger.info(f"{'=' * 70}")

            # Clone config with this seed
            seed_config = EvalConfig(
                cache_fill_ratio=config.cache_fill_ratio,
                index_type=config.index_type,
                index_params=config.index_params,
                thresholds=config.thresholds,
                min_quality=config.min_quality,
                calibration_ratio=config.calibration_ratio,
                llm_latency_ms=config.llm_latency_ms,
                llm_cost_usd=config.llm_cost_usd,
                fill_strategies=config.fill_strategies,
                seed=seed,
            )

            evaluator = cls(embeddings, texts, seed_config, metadata)
            result = evaluator.run()
            per_seed_results[str(seed)] = result

        # Aggregate across seeds
        aggregated = cls._aggregate_seeds(per_seed_results, config.fill_strategies)

        return {
            "multi_seed": True,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "aggregated": aggregated,
            "per_seed": per_seed_results,
        }

    @staticmethod
    def _aggregate_seeds(
        per_seed: dict, strategies: list[str]
    ) -> dict:
        """Compute mean ± std across seeds for key metrics."""
        agg = {}
        for strategy in strategies:
            # Collect adaptive routing metrics from each seed
            metrics_keys = [
                "test_hit_rate", "cosine_quality", "recall_at_1",
                "distance_quality", "best_threshold",
                "latency_saving_pct", "monetary_saving_pct",
            ]
            seed_metrics = {k: [] for k in metrics_keys}

            for seed_key, result in per_seed.items():
                if strategy not in result:
                    continue
                adaptive = result[strategy].get("adaptive", {})
                for k in metrics_keys:
                    if k in adaptive:
                        seed_metrics[k].append(adaptive[k])

            # Compute mean ± std
            summary = {}
            for k, vals in seed_metrics.items():
                if vals:
                    arr = np.array(vals)
                    summary[k] = {
                        "mean": round(float(np.mean(arr)), 4),
                        "std": round(float(np.std(arr)), 4),
                        "min": round(float(np.min(arr)), 4),
                        "max": round(float(np.max(arr)), 4),
                    }
            agg[strategy] = summary

        return agg

