"""
Profiler — time and memory measurement utilities for index benchmarks.

Uses time.perf_counter_ns() for nanosecond precision latency and
psutil for process-level RSS memory tracking.
"""

import gc
import logging
import time

import numpy as np
import psutil

logger = logging.getLogger(__name__)


def measure_build(index, vectors: np.ndarray) -> dict:
    """
    Build an index and measure wall-clock time and peak memory.

    Args:
        index: A BaseIndex instance (not yet built).
        vectors: Database vectors to build the index on.

    Returns:
        Dict with 'build_time_s', 'peak_rss_mb', and 'post_build_rss_mb'.
    """
    import os
    import resource

    process = psutil.Process()

    # Force GC before measuring
    gc.collect()
    
    # Get current memory before build
    mem_before = process.memory_info().rss
    
    # Reset resource maxrss tracking if possible (often not possible to reset, but we take a baseline)
    ru_before = resource.getrusage(resource.RUSAGE_SELF)

    t0 = time.perf_counter()
    index.build(vectors)
    build_time = time.perf_counter() - t0

    gc.collect()
    mem_after = process.memory_info().rss
    ru_after = resource.getrusage(resource.RUSAGE_SELF)

    # On Linux, maxrss is in kilobytes. On macOS, it's in bytes.
    # We can handle this by checking OS type, but a robust way is to estimate based on magnitude.
    # However, standard practice: macOS = bytes, Linux = KB.
    import sys
    maxrss_final = ru_after.ru_maxrss
    if sys.platform.startswith("linux"):
        maxrss_final *= 1024  # Linux: KB -> Bytes

    # Peak RSS observed. Note: it's the high water mark for the whole process lifetime up to this point
    # We try to infer if the build caused a new peak
    peak_rss_bytes = maxrss_final
    
    # Memory retained after GC (approximate cost of the index in memory)
    post_build_rss_bytes = mem_after
    
    # Calculated delta
    memory_delta = max(0, mem_after - mem_before)

    # Use the larger of: measured delta or index's self-reported estimate
    index_estimate = index.memory_usage_bytes
    memory_used = max(memory_delta, index_estimate)

    result = {
        "build_time_s": round(build_time, 4),
        "post_build_rss_mb": round(post_build_rss_bytes / (1024 * 1024), 2),
        "peak_rss_mb": round(peak_rss_bytes / (1024 * 1024), 2),
        "memory_mb": round(memory_used / (1024 * 1024), 2), # Legacy compat
        "memory_bytes": memory_used,
    }

    logger.info(
        f"Build: {build_time:.3f}s, "
        f"post_build_rss={result['post_build_rss_mb']:.1f}MB, "
        f"peak_rss={result['peak_rss_mb']:.1f}MB, "
        f"est_used={result['memory_mb']:.1f}MB "
        f"({index.ntotal} vectors)"
    )
    return result


def measure_search_latency(
    index, queries: np.ndarray, k: int, warmup: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure per-query search latency with individual timing.

    Performs warmup queries first, then measures each query individually
    using perf_counter_ns for nanosecond precision.

    Args:
        index: A built BaseIndex instance.
        queries: Query vectors, shape (Q, D).
        k: Number of nearest neighbors.
        warmup: Number of warmup queries (not timed).

    Returns:
        Tuple of (latencies_ns, all_distances, all_indices, warmup_used):
        - latencies_ns: Shape (Q,) — per-query latency in nanoseconds
        - all_distances: Shape (Q, k) — distances from each query
        - all_indices: Shape (Q, k) — neighbor IDs for each query
        - warmup_used: Int — number of queries used for warmup
    """
    Q = queries.shape[0]

    # Warmup — run a few searches to warm caches
    if warmup > 0:
        warmup_q = queries[:min(warmup, Q)]
        index.search(warmup_q, k)

    # Measure each query individually for latency distribution
    latencies_ns = np.empty(Q, dtype=np.int64)
    all_distances = np.empty((Q, k), dtype=np.float32)
    all_indices = np.empty((Q, k), dtype=np.int64)

    for i in range(Q):
        q = queries[i : i + 1]  # Shape (1, D)
        t0 = time.perf_counter_ns()
        D, I = index.search(q, k)
        latencies_ns[i] = time.perf_counter_ns() - t0
        all_distances[i] = D[0]
        all_indices[i] = I[0]

    return latencies_ns, all_distances, all_indices, warmup


def measure_batch_throughput(
    index, queries: np.ndarray, k: int, n_repeats: int = 3
) -> dict:
    """
    Measure batch search throughput (all queries at once).

    Args:
        index: A built BaseIndex instance.
        queries: Query vectors, shape (Q, D).
        k: Number of nearest neighbors.
        n_repeats: Number of times to repeat the batch for averaging.

    Returns:
        Dict with 'batch_time_s', 'throughput_qps', 'n_queries'.
    """
    Q = queries.shape[0]

    # Warmup
    index.search(queries[:min(100, Q)], k)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        index.search(queries, k)
        times.append(time.perf_counter() - t0)

    avg_time = float(np.mean(times))
    throughput = Q / avg_time if avg_time > 0 else float("inf")

    return {
        "batch_time_s": round(avg_time, 4),
        "throughput_qps": round(throughput, 1),
        "n_queries": Q,
    }
