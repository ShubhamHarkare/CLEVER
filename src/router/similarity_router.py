"""
Similarity-based query router.

Routes queries to cache or LLM based on nearest-neighbor distance
from the semantic cache. Uses L2 distance thresholds to decide
whether a cached response is "close enough."

Design note: FAISS returns L2 *squared* distances for IndexFlatL2.
For normalized vectors (norm=1), L2² = 2(1 - cosine_sim).
So distance=0 means identical, distance=2 means orthogonal.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.router.cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of a single routing decision."""

    action: str  # "cache" or "llm"
    distance: float  # L2² distance to nearest cached query
    confidence: float  # 1.0 - distance/max_distance (higher = more confident)
    cache_index: int  # Index of the cached entry (-1 if miss)
    latency_ms: float  # Time taken for the routing decision
    threshold: float  # Threshold used for this decision


class SimilarityRouter:
    """Threshold-based query router using L2 distance.

    Given a query embedding, searches the cache index for the nearest
    neighbor. If the distance is below the threshold, routes to cache;
    otherwise routes to LLM.

    Attributes:
        threshold: Maximum L2² distance for a cache hit.
        cost_model: Cost model for routing decisions.
        max_distance: Maximum meaningful distance (2.0 for normalized vectors).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        cost_model: Optional[CostModel] = None,
        max_distance: float = 2.0,
    ):
        """
        Args:
            threshold: L2² distance threshold. Queries with nearest-neighbor
                       distance below this are routed to cache.
                       For normalized embeddings: 0.0 = exact match, 2.0 = orthogonal.
                       Typical useful range: [0.1, 1.0].
            cost_model: Cost model for calculating savings.
            max_distance: Maximum possible distance (2.0 for unit-norm L2²).
        """
        self.threshold = threshold
        self.cost_model = cost_model or CostModel()
        self.max_distance = max_distance

        # Statistics
        self._n_queries = 0
        self._n_hits = 0
        self._distances: list[float] = []
        self._decisions: list[RoutingDecision] = []

    def route(
        self,
        query_embedding: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> RoutingDecision:
        """Make a routing decision for a single query.

        Args:
            query_embedding: Query vector, shape (D,) or (1, D).
            distances: L2² distances from FAISS search, shape (1, k).
            indices: Indices from FAISS search, shape (1, k).

        Returns:
            RoutingDecision with action, distance, confidence.
        """
        t_start = time.perf_counter()

        # Get nearest neighbor distance
        min_distance = float(distances[0, 0])
        nearest_idx = int(indices[0, 0])

        # Invalid result (empty index)
        if nearest_idx < 0:
            min_distance = float("inf")

        # Compute confidence: 1.0 = exact match, 0.0 = max distance
        confidence = max(0.0, 1.0 - min_distance / self.max_distance)

        # Route decision
        is_hit = min_distance <= self.threshold
        action = "cache" if is_hit else "llm"
        cache_idx = nearest_idx if is_hit else -1

        latency_ms = (time.perf_counter() - t_start) * 1000

        decision = RoutingDecision(
            action=action,
            distance=min_distance,
            confidence=confidence,
            cache_index=cache_idx,
            latency_ms=latency_ms,
            threshold=self.threshold,
        )

        # Update statistics
        self._n_queries += 1
        if is_hit:
            self._n_hits += 1
        self._distances.append(min_distance)
        self._decisions.append(decision)

        return decision

    def route_batch(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> list[RoutingDecision]:
        """Make routing decisions for a batch of queries.

        Args:
            distances: L2² distances, shape (N, k).
            indices: Indices, shape (N, k).

        Returns:
            List of RoutingDecisions.
        """
        decisions = []
        for i in range(distances.shape[0]):
            d = distances[i:i+1]
            idx = indices[i:i+1]
            dec = self.route(
                query_embedding=None,  # Not needed for threshold check
                distances=d,
                indices=idx,
            )
            decisions.append(dec)
        return decisions

    @property
    def hit_rate(self) -> float:
        """Current cache hit rate."""
        return self._n_hits / self._n_queries if self._n_queries > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Return routing statistics."""
        distances = np.array(self._distances) if self._distances else np.array([0.0])
        return {
            "n_queries": self._n_queries,
            "n_hits": self._n_hits,
            "n_misses": self._n_queries - self._n_hits,
            "hit_rate": round(self.hit_rate, 4),
            "threshold": self.threshold,
            "avg_distance": round(float(distances.mean()), 4),
            "median_distance": round(float(np.median(distances)), 4),
            "min_distance": round(float(distances.min()), 4),
            "max_distance_seen": round(float(distances.max()), 4),
        }

    def reset_stats(self):
        """Reset routing statistics."""
        self._n_queries = 0
        self._n_hits = 0
        self._distances = []
        self._decisions = []
