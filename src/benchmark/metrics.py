"""
Benchmark metrics — recall@k, latency statistics, throughput.

All functions are stateless and operate on numpy arrays.
"""

import numpy as np


def compute_recall_at_k(
    approx_ids: np.ndarray, exact_ids: np.ndarray, k: int
) -> float:
    """
    Compute recall@k: fraction of true top-k results found by approximate search.

    Args:
        approx_ids: Shape (Q, k) — IDs returned by approximate index.
        exact_ids: Shape (Q, k) — IDs returned by exact (Flat) index.
        k: Number of neighbors (must match second dim of both arrays).

    Returns:
        Average recall@k across all queries (float in [0, 1]).
    """
    assert approx_ids.shape == exact_ids.shape, (
        f"Shape mismatch: approx={approx_ids.shape}, exact={exact_ids.shape}"
    )
    Q = approx_ids.shape[0]
    total_recall = 0.0

    for i in range(Q):
        approx_set = set(approx_ids[i, :k])
        exact_set = set(exact_ids[i, :k])
        # Remove -1 (FAISS returns -1 for missing neighbors)
        approx_set.discard(-1)
        exact_set.discard(-1)
        if len(exact_set) == 0:
            total_recall += 1.0  # No ground truth → trivially correct
        else:
            total_recall += len(approx_set & exact_set) / len(exact_set)

    return total_recall / Q


def compute_latency_stats(latencies_ns: np.ndarray) -> dict:
    """
    Compute latency statistics from per-query nanosecond timings.

    Args:
        latencies_ns: Array of per-query latencies in nanoseconds.

    Returns:
        Dict with mean, std, p50, p95, p99 latencies in milliseconds.
    """
    latencies_ms = latencies_ns / 1e6  # ns → ms
    return {
        "mean": float(np.mean(latencies_ms)),
        "std": float(np.std(latencies_ms)),
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
    }


def compute_throughput(total_time_s: float, n_queries: int) -> float:
    """
    Compute throughput in queries per second.

    Args:
        total_time_s: Total wall-clock batch search time in seconds.
        n_queries: Number of queries in the batch.

    Returns:
        Throughput in queries/second.
    """
    if total_time_s <= 0:
        return float("inf")
    return n_queries / total_time_s
