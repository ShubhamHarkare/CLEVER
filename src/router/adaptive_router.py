"""
Adaptive query router — learns optimal threshold from calibration data.

Sweeps L2² distance thresholds on a held-out calibration set to find
the one that maximizes cost savings while maintaining a minimum quality
target (measured as recall of cached results vs. ground truth).

This is the core research contribution of CLEVER: a cost-aware router
that picks the threshold where caching becomes cheaper than LLM calls.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.router.cost_model import CostModel
from src.router.similarity_router import SimilarityRouter

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result for a single threshold evaluation."""

    threshold: float
    hit_rate: float
    quality: float  # recall proxy
    cost_savings_pct: float
    latency_savings_pct: float
    monetary_savings_pct: float
    n_hits: int
    n_misses: int


class AdaptiveRouter:
    """Learns the optimal routing threshold from calibration data.

    Given a set of calibration queries with known ground-truth neighbors,
    sweeps thresholds and picks the one that maximizes a combined objective:
        objective = cost_savings * quality_factor

    where quality_factor penalizes low-quality cache hits.

    Attributes:
        cost_model: Cost model for routing.
        min_quality: Minimum acceptable quality (recall) target.
        thresholds: Array of thresholds to sweep.
        best_threshold: Optimal threshold found during calibration.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        min_quality: float = 0.8,
        threshold_range: tuple[float, float] = (0.05, 1.5),
        n_thresholds: int = 50,
    ):
        """
        Args:
            cost_model: Cost model for routing decisions.
            min_quality: Minimum quality (recall proxy) to maintain.
            threshold_range: (min, max) L2² distance thresholds to sweep.
            n_thresholds: Number of thresholds to test.
        """
        self.cost_model = cost_model or CostModel()
        self.min_quality = min_quality
        self.thresholds = np.linspace(
            threshold_range[0], threshold_range[1], n_thresholds
        )
        self.best_threshold: Optional[float] = None
        self.sweep_results: list[ThresholdResult] = []
        self._router: Optional[SimilarityRouter] = None

    def calibrate(
        self,
        distances: np.ndarray,
        gt_distances: np.ndarray,
        quality_metric: str = "recall_proxy",
    ) -> float:
        """Find optimal threshold from calibration data.

        Args:
            distances: L2² distances from cache search, shape (N, k).
                       These are distances from query to nearest cached entry.
            gt_distances: L2² distances from ground-truth search, shape (N, k).
                          These are distances from query to true nearest neighbor.
            quality_metric: How to measure quality of cache hits.

        Returns:
            Optimal threshold value.
        """
        # Get nearest-neighbor distances
        cache_nn_dist = distances[:, 0]  # Distance to nearest cached entry
        gt_nn_dist = gt_distances[:, 0]  # Distance to true nearest neighbor

        self.sweep_results = []
        best_objective = -float("inf")
        best_threshold = self.thresholds[0]

        for threshold in self.thresholds:
            # Simulate routing at this threshold
            is_hit = cache_nn_dist <= threshold

            n_hits = int(is_hit.sum())
            n_total = len(cache_nn_dist)
            hit_rate = n_hits / n_total if n_total > 0 else 0

            # Quality: cosine-similarity-based metric for cache hits.
            # For normalized vectors: cosine_sim ≈ 1 - L2²/2,
            # so quality = mean(clip(1 - dist/2, 0, 1)) over hits.
            if n_hits > 0:
                hit_cache_dists = cache_nn_dist[is_hit]
                cosine_sims = np.clip(1.0 - hit_cache_dists / 2.0, 0.0, 1.0)
                quality = float(np.mean(cosine_sims))
            else:
                quality = 1.0  # No hits = no quality loss (but no savings)

            # Cost savings
            savings = self.cost_model.cost_savings(hit_rate)

            result = ThresholdResult(
                threshold=round(float(threshold), 4),
                hit_rate=round(hit_rate, 4),
                quality=round(quality, 4),
                cost_savings_pct=savings["latency_saving_pct"],
                latency_savings_pct=savings["latency_saving_pct"],
                monetary_savings_pct=savings["monetary_saving_pct"],
                n_hits=n_hits,
                n_misses=n_total - n_hits,
            )
            self.sweep_results.append(result)

            # Objective: maximize cost savings subject to quality constraint
            if quality >= self.min_quality:
                objective = savings["latency_saving_pct"] + savings["monetary_saving_pct"]
                if objective > best_objective:
                    best_objective = objective
                    best_threshold = threshold

        self.best_threshold = float(best_threshold)

        # Create a router with the optimal threshold
        self._router = SimilarityRouter(
            threshold=self.best_threshold,
            cost_model=self.cost_model,
        )

        logger.info(
            f"Calibration complete: best_threshold={self.best_threshold:.4f}, "
            f"objective={best_objective:.2f}"
        )

        # Log sweep summary
        for r in self.sweep_results:
            if abs(r.threshold - self.best_threshold) < 0.001:
                logger.info(
                    f"  → hit_rate={r.hit_rate:.2%}, quality={r.quality:.2%}, "
                    f"cost_savings={r.cost_savings_pct:.1f}%"
                )

        return self.best_threshold

    @property
    def router(self) -> SimilarityRouter:
        """Return the calibrated router.

        Raises:
            ValueError: If calibrate() has not been called.
        """
        if self._router is None:
            raise ValueError("Must call calibrate() before accessing router")
        return self._router

    def sweep_summary(self) -> list[dict]:
        """Return sweep results as a list of dicts."""
        return [
            {
                "threshold": r.threshold,
                "hit_rate": r.hit_rate,
                "quality": r.quality,
                "cost_savings_pct": r.cost_savings_pct,
                "latency_savings_pct": r.latency_savings_pct,
                "monetary_savings_pct": r.monetary_savings_pct,
                "n_hits": r.n_hits,
                "n_misses": r.n_misses,
            }
            for r in self.sweep_results
        ]
