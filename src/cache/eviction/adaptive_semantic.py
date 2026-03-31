"""
Adaptive semantic eviction policy.

Extends SemanticPolicy by monitoring the incoming query stream's
diversity and adjusting the recency-vs-redundancy balance online.

Key insight: Redundancy-based eviction helps when queries are
clustered (many near-duplicates waste cache slots). But when
queries are diverse/uniform, there's no redundancy to exploit
and LRU-style recency is the better signal.

This policy detects the regime automatically:
  - High diversity (uniform traffic) → increase α (recency weight),
    decrease redundancy influence → behaves like LRU
  - Low diversity (clustered traffic) → decrease α, let redundancy
    dominate → aggressive semantic eviction

Adaptation uses the normalized cluster entropy of a sliding window
of recent query embeddings.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from src.cache.eviction.semantic import SemanticPolicy

logger = logging.getLogger(__name__)


class AdaptiveSemanticPolicy(SemanticPolicy):
    """Semantic eviction with online workload adaptation.

    Monitors a sliding window of recent query embeddings, periodically
    computes their diversity, and adjusts α (recency weight) accordingly.

    When diversity is high → α is high (recency-dominated, LRU-like).
    When diversity is low → α is low (redundancy-dominated, semantic).

    The β (frequency weight) stays fixed — frequency is always useful.

    Args:
        similarity_threshold: L2² distance threshold for neighbours.
        base_alpha: Starting α value (will be adapted).
        beta: Fixed frequency weight.
        recompute_interval: Evictions between redundancy recomputation.
        adaptation_window: Number of recent queries to track.
        adaptation_interval: Queries between diversity re-estimation.
        alpha_min: Minimum α (most redundancy-focused).
        alpha_max: Maximum α (most recency-focused / LRU-like).
        n_clusters: Clusters for diversity estimation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        base_alpha: float = 1.0,
        beta: float = 1.0,
        recompute_interval: int = 50,
        adaptation_window: int = 500,
        adaptation_interval: int = 200,
        alpha_min: float = 0.3,
        alpha_max: float = 2.5,
        n_clusters: int = 20,
    ) -> None:
        super().__init__(
            similarity_threshold=similarity_threshold,
            alpha=base_alpha,
            beta=beta,
            recompute_interval=recompute_interval,
        )
        self.adaptation_window = adaptation_window
        self.adaptation_interval = adaptation_interval
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._adapt_n_clusters = n_clusters

        # Sliding window of recent query embeddings
        self._recent_queries: deque[np.ndarray] = deque(maxlen=adaptation_window)
        self._queries_since_adapt: int = 0
        self._n_adaptations: int = 0

        # Track alpha trajectory for analysis
        self._alpha_history: list[dict] = []
        self._current_diversity: float = 0.5  # initial estimate

    def on_query(self, query_embedding: np.ndarray) -> None:
        """Record an incoming query for diversity tracking.

        This must be called by the eviction runner for every query
        in the stream (both hits and misses), BEFORE on_access or
        on_insert.
        """
        self._recent_queries.append(query_embedding.copy())
        self._queries_since_adapt += 1

        if self._queries_since_adapt >= self.adaptation_interval:
            self._adapt_alpha()
            self._queries_since_adapt = 0

    def _adapt_alpha(self) -> None:
        """Recompute diversity and adjust alpha."""
        if len(self._recent_queries) < 50:
            return  # not enough data yet

        # Stack recent embeddings
        window = np.array(list(self._recent_queries), dtype=np.float32)

        # Compute diversity via cluster entropy
        from src.benchmark.workload import compute_workload_diversity
        diversity = compute_workload_diversity(
            window,
            n_clusters=self._adapt_n_clusters,
            seed=42 + self._n_adaptations,
        )

        self._current_diversity = diversity["cluster_entropy"]

        # Linear interpolation: high diversity → high alpha (LRU-like)
        #                        low diversity → low alpha (redundancy-focused)
        old_alpha = self.alpha
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * self._current_diversity

        self._n_adaptations += 1
        self._alpha_history.append({
            "adaptation_step": self._n_adaptations,
            "n_queries_seen": len(self._recent_queries),
            "diversity": round(self._current_diversity, 4),
            "effective_clusters": diversity["effective_clusters"],
            "old_alpha": round(old_alpha, 4),
            "new_alpha": round(self.alpha, 4),
        })

        if self._n_adaptations <= 5 or self._n_adaptations % 10 == 0:
            logger.info(
                f"[adaptive] step={self._n_adaptations}: "
                f"diversity={self._current_diversity:.3f}, "
                f"α: {old_alpha:.2f} → {self.alpha:.2f}"
            )

    # ── Stats override ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return base stats plus adaptation history."""
        base = super().stats
        base.update({
            "n_adaptations": self._n_adaptations,
            "current_diversity": round(self._current_diversity, 4),
            "current_alpha": round(self.alpha, 4),
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "adaptation_window": self.adaptation_window,
            "adaptation_interval": self.adaptation_interval,
            "alpha_history": self._alpha_history,
        })
        return base

    @property
    def name(self) -> str:
        return "adaptive"

    def __repr__(self) -> str:
        return (
            f"AdaptiveSemanticPolicy("
            f"threshold={self.similarity_threshold}, "
            f"α={self.alpha:.2f} [{self.alpha_min}, {self.alpha_max}], "
            f"β={self.beta}, "
            f"diversity={self._current_diversity:.3f}, "
            f"adaptations={self._n_adaptations})"
        )