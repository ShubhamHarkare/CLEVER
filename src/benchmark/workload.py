"""
Workload generator — create realistic query streams for benchmarking.

Supports three workload patterns matching real LLM cache usage:
- uniform: random query selection (baseline)
- clustered: 80/20 Zipf over k-means topic clusters
- bursty: temporal locality with burst patterns
"""

import logging

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


def generate_workload(
    query_vectors: np.ndarray,
    db_vectors: np.ndarray,
    workload_type: str,
    n_queries: int,
    seed: int = 42,
    n_clusters: int = 50,
) -> np.ndarray:
    """
    Generate a query workload (indices into the query_vectors array).

    Args:
        query_vectors: Held-out query pool embeddings, shape (Q, D).
        db_vectors: Database embeddings for clustering context, shape (N, D).
        workload_type: One of "uniform", "clustered", "bursty".
        n_queries: Number of query indices to generate.
        seed: Random seed for reproducibility.
        n_clusters: Number of topic clusters (for clustered/bursty).

    Returns:
        Array of indices into query_vectors, shape (n_queries,).
    """
    workload_type = workload_type.lower().strip()

    if workload_type == "uniform":
        return _uniform_workload(query_vectors, n_queries, seed)
    elif workload_type == "clustered":
        return _clustered_workload(query_vectors, db_vectors, n_queries, seed, n_clusters)
    elif workload_type == "bursty":
        return _bursty_workload(query_vectors, db_vectors, n_queries, seed, n_clusters)
    else:
        raise ValueError(
            f"Unknown workload type '{workload_type}'. "
            f"Available: uniform, clustered, bursty"
        )


def _uniform_workload(
    query_vectors: np.ndarray, n_queries: int, seed: int
) -> np.ndarray:
    """Random uniform selection — baseline workload."""
    rng = np.random.RandomState(seed)
    Q = query_vectors.shape[0]
    return rng.choice(Q, size=n_queries, replace=True)


def _clustered_workload(
    query_vectors: np.ndarray, db_vectors: np.ndarray, n_queries: int, seed: int, n_clusters: int
) -> np.ndarray:
    """
    Zipf-distributed workload over topic clusters.

    80% of queries come from 20% of clusters (power-law distribution).
    Simulates real LLM usage where some topics are much more popular.
    """
    rng = np.random.RandomState(seed)
    N = db_vectors.shape[0]
    Q = query_vectors.shape[0]

    # Adjust n_clusters if dataset is small
    actual_k = min(n_clusters, N // 5)
    if actual_k < 2:
        actual_k = 2

    # Cluster db_vectors to define the concept space
    logger.info(f"Clustering {N} DB vectors into {actual_k} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=actual_k, random_state=seed, batch_size=min(1024, N),
        n_init=3
    )
    kmeans.fit(db_vectors)
    
    # Assign query_vectors to those clusters
    labels = kmeans.predict(query_vectors)

    # Build cluster-to-query-indices mapping
    cluster_indices = {}
    for i in range(actual_k):
        members = np.where(labels == i)[0]
        if len(members) > 0:
            cluster_indices[i] = members

    # Zipf distribution over valid clusters (power-law)
    cluster_ids = sorted(cluster_indices.keys())
    if not cluster_ids:
        logger.warning("No valid clusters found for queries. Falling back to uniform.")
        return rng.choice(Q, size=n_queries, replace=True)

    ranks = np.arange(1, len(cluster_ids) + 1, dtype=np.float64)
    weights = 1.0 / ranks  # Zipf: P(rank r) ∝ 1/r
    weights /= weights.sum()

    # Sample queries
    query_indices = np.empty(n_queries, dtype=np.int64)
    chosen_clusters = rng.choice(cluster_ids, size=n_queries, p=weights)

    for i, c in enumerate(chosen_clusters):
        members = cluster_indices[c]
        query_indices[i] = rng.choice(members)

    return query_indices


def _bursty_workload(
    query_vectors: np.ndarray, db_vectors: np.ndarray, n_queries: int, seed: int, n_clusters: int
) -> np.ndarray:
    """
    Bursty workload with temporal locality.

    Queries arrive in bursts from the same topic cluster, then switch
    to a different cluster. Simulates real-world patterns where users
    explore one topic for a while, then switch.

    Burst length is drawn from a geometric distribution (mean ~20 queries).
    """
    rng = np.random.RandomState(seed)
    N = db_vectors.shape[0]
    Q = query_vectors.shape[0]

    actual_k = min(n_clusters, N // 5)
    if actual_k < 2:
        actual_k = 2

    # Cluster db_vectors to define the concept space
    logger.info(f"Clustering {N} vectors into {actual_k} clusters for bursty workload...")
    kmeans = MiniBatchKMeans(
        n_clusters=actual_k, random_state=seed, batch_size=min(1024, N),
        n_init=3
    )
    kmeans.fit(db_vectors)

    # Assign query_vectors
    labels = kmeans.predict(query_vectors)

    cluster_indices = {}
    for i in range(actual_k):
        members = np.where(labels == i)[0]
        if len(members) > 0:
            cluster_indices[i] = members

    cluster_ids = list(cluster_indices.keys())
    if not cluster_ids:
        logger.warning("No valid clusters found for queries. Falling back to uniform.")
        return rng.choice(Q, size=n_queries, replace=True)
    query_indices = []
    mean_burst_length = 20

    while len(query_indices) < n_queries:
        # Pick a random cluster for this burst
        c = rng.choice(cluster_ids)
        members = cluster_indices[c]

        # Geometric distribution for burst length
        burst_len = rng.geometric(1.0 / mean_burst_length)
        burst_len = min(burst_len, n_queries - len(query_indices))

        # Sample from this cluster
        burst = rng.choice(members, size=burst_len, replace=True)
        query_indices.extend(burst.tolist())

    return np.array(query_indices[:n_queries], dtype=np.int64)
