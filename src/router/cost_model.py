"""
Cost model for query routing decisions.

Parameterizes the cost of serving a query from the semantic cache
vs. forwarding it to an LLM. Costs can be auto-populated from
Phase 2 benchmark results.

Key equation:
    route_to_cache  iff  cache_cost + miss_risk * llm_cost  <  llm_cost
    i.e.,  cache_cost  <  llm_cost * (1 - miss_risk)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Parameterizes costs for cache-vs-LLM routing decisions.

    Attributes:
        llm_latency_ms: Average LLM response latency.
        llm_cost_usd: Monetary cost per LLM query.
        llm_quality: Quality score of LLM responses (reference = 1.0).
        cache_latency_ms: Average cache lookup latency.
        cache_cost_usd: Monetary cost per cache lookup (≈0 for local).
        cache_build_time_s: Time to build the cache index.
        cache_memory_mb: Memory used by the cache index.
        index_type: FAISS index type backing the cache.
        index_params: Parameters for the index.
    """

    # LLM costs (defaults based on typical API costs)
    llm_latency_ms: float = 500.0
    llm_cost_usd: float = 0.01
    llm_quality: float = 1.0

    # Cache costs (populated from benchmarks or manually)
    cache_latency_ms: float = 1.0
    cache_cost_usd: float = 0.0
    cache_build_time_s: float = 0.0
    cache_memory_mb: float = 0.0

    # Index configuration
    index_type: str = "hnsw"
    index_params: dict = field(default_factory=lambda: {
        "M": 32, "efConstruction": 128, "efSearch": 128,
    })

    def routing_cost(self, is_cache_hit: bool) -> float:
        """Compute the total cost for a routing decision.

        Args:
            is_cache_hit: Whether the cache returned a valid result.

        Returns:
            Total cost (latency-weighted + monetary) in abstract units.
        """
        if is_cache_hit:
            # Cache hit: only cache lookup cost
            return self.cache_latency_ms + self.cache_cost_usd * 1000
        else:
            # Cache miss: cache lookup + LLM call
            return (
                self.cache_latency_ms
                + self.llm_latency_ms
                + self.llm_cost_usd * 1000
            )

    def direct_llm_cost(self) -> float:
        """Cost of always going to the LLM (no cache)."""
        return self.llm_latency_ms + self.llm_cost_usd * 1000

    def cost_savings(self, hit_rate: float) -> dict:
        """Calculate cost savings for a given cache hit rate.

        Args:
            hit_rate: Fraction of queries served from cache [0, 1].

        Returns:
            Dict with latency savings, monetary savings, and total savings.
        """
        baseline_latency = self.llm_latency_ms
        actual_latency = (
            hit_rate * self.cache_latency_ms
            + (1 - hit_rate) * self.llm_latency_ms
        )
        latency_saving_pct = (
            (baseline_latency - actual_latency) / baseline_latency * 100
            if baseline_latency > 0 else 0
        )

        baseline_monetary = self.llm_cost_usd
        actual_monetary = (1 - hit_rate) * self.llm_cost_usd
        monetary_saving_pct = (
            (baseline_monetary - actual_monetary) / baseline_monetary * 100
            if baseline_monetary > 0 else 0
        )

        return {
            "hit_rate": hit_rate,
            "avg_latency_ms": actual_latency,
            "latency_saving_pct": round(latency_saving_pct, 2),
            "avg_cost_usd": actual_monetary,
            "monetary_saving_pct": round(monetary_saving_pct, 2),
            "cache_overhead_memory_mb": self.cache_memory_mb,
            "cache_overhead_build_s": self.cache_build_time_s,
        }

    @classmethod
    def from_benchmark(
        cls,
        benchmark_path: str | Path,
        index_type: str = "hnsw",
        llm_latency_ms: float = 500.0,
        llm_cost_usd: float = 0.01,
    ) -> "CostModel":
        """Create a CostModel from Phase 2 benchmark results.

        Picks the best-recall configuration for the given index type.

        Args:
            benchmark_path: Path to a benchmark JSON file.
            index_type: Index type to use ("hnsw", "ivf", etc.).
            llm_latency_ms: LLM latency assumption.
            llm_cost_usd: LLM cost assumption.
        """
        benchmark_path = Path(benchmark_path)
        with open(benchmark_path) as f:
            results = json.load(f)

        # Filter to the requested index type
        entries = [r for r in results if r["index_type"] == index_type and "error" not in r]

        if not entries:
            logger.warning(
                f"No benchmark results for index_type='{index_type}' in "
                f"{benchmark_path}. Using defaults."
            )
            return cls(
                llm_latency_ms=llm_latency_ms,
                llm_cost_usd=llm_cost_usd,
                index_type=index_type,
            )

        # Pick the best-recall configuration
        best = max(entries, key=lambda r: r.get("recall_at_10", 0))
        params = best.get("params", {})

        model = cls(
            llm_latency_ms=llm_latency_ms,
            llm_cost_usd=llm_cost_usd,
            cache_latency_ms=best["search_latency_ms"]["p50"],
            cache_build_time_s=best.get("build_time_s", 0),
            cache_memory_mb=best.get("memory_mb", 0),
            index_type=index_type,
            index_params=params,
        )

        logger.info(
            f"CostModel from benchmark: index={index_type}, "
            f"params={params}, "
            f"cache_latency={model.cache_latency_ms:.2f}ms, "
            f"recall@10={best.get('recall_at_10', 'N/A')}"
        )
        return model

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"CostModel(\n"
            f"  LLM:   latency={self.llm_latency_ms}ms, cost=${self.llm_cost_usd}/query\n"
            f"  Cache:  latency={self.cache_latency_ms:.2f}ms, "
            f"mem={self.cache_memory_mb:.0f}MB, "
            f"build={self.cache_build_time_s:.1f}s\n"
            f"  Index:  {self.index_type} {self.index_params}\n"
            f")"
        )
