"""
Semantic cache backed by a FAISS index.

Stores query embeddings alongside their text and (simulated) responses.
Supports lookup (search for nearest cached entry), insert (add new
entry), and statistics tracking.

For CLEVER, we don't actually call an LLM — we simulate responses.
The cache stores the original query text as the "response" and
measures whether the router would have served it correctly.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.indexes.factory import create_index
from src.indexes.base import BaseIndex

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single entry in the semantic cache."""

    cache_id: int
    query_text: str
    embedding: np.ndarray  # shape (D,)
    response: str  # Simulated or actual response
    metadata: dict = field(default_factory=dict)


@dataclass
class LookupResult:
    """Result of a cache lookup operation."""

    hit: bool
    distance: float  # L2² distance to nearest entry
    cache_entry: Optional[CacheEntry]
    lookup_time_ms: float
    k_distances: np.ndarray  # All k distances
    k_indices: np.ndarray  # All k indices


class SemanticCache:
    """FAISS-backed semantic cache for query deduplication.

    Wraps a FAISS index and maintains a mapping from index IDs to
    cache entries (query text, response, metadata).

    Attributes:
        index: The underlying FAISS index.
        entries: List of cached entries.
        dim: Embedding dimensionality.
    """

    def __init__(
        self,
        dim: int = 384,
        index_type: str = "hnsw",
        index_params: Optional[dict] = None,
    ):
        """
        Args:
            dim: Embedding dimensionality.
            index_type: FAISS index type ("flat", "hnsw", "ivf", "lsh").
            index_params: Parameters for the index.
        """
        self.dim = dim
        self.index_type = index_type
        self.index_params = index_params or {}
        self._index: Optional[BaseIndex] = None
        self._entries: list[CacheEntry] = []

        # Statistics
        self._n_lookups = 0
        self._n_hits = 0
        self._n_inserts = 0
        self._total_lookup_time_ms = 0.0

    def build(self, embeddings: np.ndarray, texts: list[str],
              responses: Optional[list[str]] = None,
              metadata_list: Optional[list[dict]] = None) -> float:
        """Build the cache from a set of embeddings and texts.

        Args:
            embeddings: Array of shape (N, D), dtype float32.
            texts: List of N query texts.
            responses: List of N response texts (defaults to query texts).
            metadata_list: List of N metadata dicts.

        Returns:
            Build time in seconds.
        """
        n = len(embeddings)
        if responses is None:
            responses = texts  # Use query text as simulated response
        if metadata_list is None:
            metadata_list = [{}] * n

        # Build entries
        self._entries = []
        for i in range(n):
            entry = CacheEntry(
                cache_id=i,
                query_text=texts[i],
                embedding=embeddings[i],
                response=responses[i],
                metadata=metadata_list[i],
            )
            self._entries.append(entry)

        # Build FAISS index
        t_start = time.perf_counter()
        self._index = create_index(
            self.index_type, dim=self.dim, **self.index_params
        )
        self._index.build(embeddings.astype(np.float32))
        build_time = time.perf_counter() - t_start

        self._n_inserts = n
        logger.info(
            f"SemanticCache built: {n} entries, "
            f"index={self.index_type}, "
            f"build_time={build_time:.3f}s"
        )
        return build_time

    def lookup(
        self, query_embedding: np.ndarray, k: int = 1
    ) -> LookupResult:
        """Search the cache for the nearest entry to the query.

        Args:
            query_embedding: Query vector, shape (D,) or (1, D).
            k: Number of nearest neighbors to return.

        Returns:
            LookupResult with distances and nearest cache entry.
        """
        if self._index is None:
            raise ValueError("Cache not built. Call build() first.")

        # Reshape to (1, D) if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        t_start = time.perf_counter()
        distances, indices = self._index.search(
            query_embedding.astype(np.float32), k=k
        )
        lookup_time_ms = (time.perf_counter() - t_start) * 1000

        # Get nearest entry
        nearest_idx = int(indices[0, 0])
        nearest_dist = float(distances[0, 0])

        # Check for valid result
        if nearest_idx >= 0 and nearest_idx < len(self._entries):
            entry = self._entries[nearest_idx]
            hit = True
        else:
            entry = None
            hit = False

        self._n_lookups += 1
        self._total_lookup_time_ms += lookup_time_ms

        return LookupResult(
            hit=hit,
            distance=nearest_dist,
            cache_entry=entry,
            lookup_time_ms=lookup_time_ms,
            k_distances=distances[0],
            k_indices=indices[0],
        )

    def batch_lookup(
        self, query_embeddings: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch search — returns raw distances and indices.

        More efficient than calling lookup() in a loop because
        it uses a single FAISS search call.

        Args:
            query_embeddings: Shape (N, D).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices), both shape (N, k).
        """
        if self._index is None:
            raise ValueError("Cache not built. Call build() first.")
        return self._index.search(
            query_embeddings.astype(np.float32), k=k
        )

    def insert(
        self, embedding: np.ndarray, text: str,
        response: Optional[str] = None, metadata: Optional[dict] = None
    ):
        """Insert a new entry into the cache.

        Args:
            embedding: Embedding vector, shape (D,).
            text: Query text.
            response: Response text.
            metadata: Additional metadata.
        """
        if self._index is None:
            raise ValueError("Cache not built. Call build() first.")

        cache_id = len(self._entries)
        entry = CacheEntry(
            cache_id=cache_id,
            query_text=text,
            embedding=embedding,
            response=response or text,
            metadata=metadata or {},
        )
        self._entries.append(entry)

        # Add to FAISS index
        emb = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(emb)
        self._n_inserts += 1

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._entries)

    @property
    def memory_mb(self) -> float:
        """Estimated memory usage in MB."""
        if self._index is None:
            return 0.0
        return self._index.memory_usage_bytes / (1024 * 1024)

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        avg_lookup = (
            self._total_lookup_time_ms / self._n_lookups
            if self._n_lookups > 0 else 0
        )
        return {
            "size": self.size,
            "n_lookups": self._n_lookups,
            "n_inserts": self._n_inserts,
            "index_type": self.index_type,
            "memory_mb": round(self.memory_mb, 2),
            "avg_lookup_time_ms": round(avg_lookup, 4),
        }
