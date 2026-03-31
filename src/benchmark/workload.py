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



#! Below are the additions done by sharkare
"""
ADD these two functions to the bottom of src/benchmark/workload.py
They go AFTER the existing _bursty_workload function.
No existing code needs to change.
"""


def generate_concentrated_workload(
    query_vectors: np.ndarray,
    n_queries: int,
    gamma: float = 0.5,
    seed: int = 42,
    n_clusters: int = 50,
) -> np.ndarray:
    """
    Generate a workload with tunable concentration (cluster skew).

    gamma controls how concentrated the workload is:
      - gamma=0.0: perfectly uniform across clusters (every cluster equally likely)
      - gamma=0.5: moderate Zipf skew (realistic LLM traffic)
      - gamma=1.0: extreme concentration (almost all queries from top 1-2 clusters)

    The Zipf exponent is: s = 1 + 3*gamma
      gamma=0.0 → s=1.0 (standard Zipf, mild skew)
      gamma=0.5 → s=2.5 (strong skew)
      gamma=1.0 → s=4.0 (extreme skew)

    Args:
        query_vectors: Query pool embeddings, shape (Q, D).
        n_queries: Number of query indices to generate.
        gamma: Concentration parameter in [0.0, 1.0].
        seed: Random seed.
        n_clusters: Number of topic clusters.

    Returns:
        Array of indices into query_vectors, shape (n_queries,).
    """
    rng = np.random.RandomState(seed)
    Q = query_vectors.shape[0]

    # Edge case: gamma=0 means truly uniform (no clustering needed)
    if gamma <= 0.001:
        return rng.choice(Q, size=n_queries, replace=True)

    # Cluster the query vectors themselves
    actual_k = min(n_clusters, Q // 5)
    if actual_k < 2:
        actual_k = 2

    kmeans = MiniBatchKMeans(
        n_clusters=actual_k, random_state=seed,
        batch_size=min(1024, Q), n_init=3,
    )
    labels = kmeans.fit_predict(query_vectors)

    # Build cluster -> member indices mapping
    cluster_indices = {}
    for i in range(actual_k):
        members = np.where(labels == i)[0]
        if len(members) > 0:
            cluster_indices[i] = members

    cluster_ids = sorted(cluster_indices.keys())
    if not cluster_ids:
        return rng.choice(Q, size=n_queries, replace=True)

    # Zipf with tunable exponent: s = 1 + 3*gamma
    zipf_exponent = 1.0 + 3.0 * gamma
    ranks = np.arange(1, len(cluster_ids) + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, zipf_exponent)
    weights /= weights.sum()

    # Sample clusters, then sample queries within each cluster
    query_indices = np.empty(n_queries, dtype=np.int64)
    chosen_clusters = rng.choice(cluster_ids, size=n_queries, p=weights)

    for i, c in enumerate(chosen_clusters):
        members = cluster_indices[c]
        query_indices[i] = rng.choice(members)

    return query_indices


def compute_workload_diversity(
    embeddings: np.ndarray,
    n_clusters: int = 30,
    seed: int = 42,
) -> dict:
    """
    Compute a workload diversity metric from a set of embeddings.

    Measures how spread out the queries are in embedding space.
    Returns multiple diversity indicators:

    - normalized_diversity: Primary metric in [0, 1]. Based on average
      pairwise L2² distance, normalized against a calibration baseline.
      High = uniform spread, Low = concentrated in few directions.

    - cluster_entropy: Entropy of cluster assignment distribution,
      normalized by log(n_clusters_requested).

    - effective_clusters: exp(raw entropy) — the effective number of topics.

    - avg_pairwise_distance: Mean L2² distance between sampled pairs.

    Args:
        embeddings: Query embeddings, shape (N, D).
        n_clusters: Number of clusters for entropy computation.
        seed: Random seed.

    Returns:
        Dict with diversity metrics.
    """
    N = embeddings.shape[0]
    D = embeddings.shape[1]
    rng = np.random.RandomState(seed)

    # --- Average pairwise distance (primary signal) ---
    sample_size = min(1000, N)
    idx = rng.choice(N, size=sample_size, replace=False)
    sample = embeddings[idx]

    n_pairs = min(5000, sample_size * (sample_size - 1) // 2)
    pair_i = rng.randint(0, sample_size, size=n_pairs)
    pair_j = rng.randint(0, sample_size, size=n_pairs)
    mask = pair_i != pair_j
    pair_i = pair_i[mask]
    pair_j = pair_j[mask]

    diffs = sample[pair_i] - sample[pair_j]
    pairwise_dists = np.sum(diffs ** 2, axis=1)
    avg_pairwise_distance = float(np.mean(pairwise_dists))

    # Normalize pairwise distance to [0, 1] using expected distance
    # for unit-norm random vectors in D dimensions.
    # For unit-norm vectors: E[||x-y||²] = 2.0
    # For non-normalized: scale by dimension.
    sample_norms = np.linalg.norm(sample, axis=1)
    avg_norm = float(np.mean(sample_norms))
    if avg_norm > 0:
        # Expected L2² between random vectors with this avg norm
        expected_dist = 2.0 * (avg_norm ** 2)
        normalized_diversity = min(1.0, avg_pairwise_distance / expected_dist)
    else:
        normalized_diversity = 0.0

    # --- Cluster entropy (secondary signal) ---
    actual_k = min(n_clusters, N // 5)
    if actual_k < 2:
        actual_k = 2

    kmeans = MiniBatchKMeans(
        n_clusters=actual_k, random_state=seed,
        batch_size=min(1024, N), n_init=3,
    )
    labels = kmeans.fit_predict(embeddings)

    counts = np.bincount(labels, minlength=actual_k).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(actual_k)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    effective_clusters = float(np.exp(entropy))

    # --- Combined metric: use pairwise distance as primary ---
    # cluster_entropy is reported but normalized_diversity drives adaptation
    return {
        "cluster_entropy": round(float(normalized_diversity), 4),
        "effective_clusters": round(effective_clusters, 2),
        "avg_pairwise_distance": round(avg_pairwise_distance, 4),
        "raw_entropy": round(float(normalized_entropy), 4),
        "n_clusters_used": int(len(probs)),
        "n_queries": N,
    }