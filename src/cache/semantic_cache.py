"""
Semantic cache backed by a FAISS index.

Stores query embeddings alongside their text and (simulated) responses.
Supports lookup (search for nearest cached entry), insert (add new
entry), eviction (LRU / LFU), and statistics tracking.

For CLEVER, we don't actually call an LLM — we simulate responses.
The cache stores the original query text as the "response" and
measures whether the router would have served it correctly.

Eviction strategy
-----------------
FAISS HNSW does not support ``remove_ids``.  We therefore use **logical
deletion**: evicted entries are marked inactive and filtered from search
results.  When dead entries exceed ``rebuild_threshold`` (default 20%)
of total, the FAISS index is rebuilt from scratch with only the active
entries.
"""

import logging
import time
from collections import OrderedDict
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

    Supports optional bounded-size caching with LRU or LFU eviction.

    Attributes:
        index: The underlying FAISS index.
        entries: List of cached entries.
        dim: Embedding dimensionality.
        max_size: Maximum cache entries (0 = unbounded).
        eviction_policy: ``"none"`` | ``"lru"`` | ``"lfu"``.
    """

    def __init__(
        self,
        dim: int = 384,
        index_type: str = "hnsw",
        index_params: Optional[dict] = None,
        max_size: int = 0,
        eviction_policy: str = "none",
        rebuild_threshold: float = 0.20,
    ):
        """
        Args:
            dim: Embedding dimensionality.
            index_type: FAISS index type ("flat", "hnsw", "ivf", "lsh").
            index_params: Parameters for the index.
            max_size: Maximum number of active cache entries.
                      0 means unbounded (no eviction).
            eviction_policy: ``"none"`` (unbounded), ``"lru"``, or ``"lfu"``.
            rebuild_threshold: Fraction of dead entries that triggers a
                               full index rebuild (default 0.20).
        """
        self.dim = dim
        self.index_type = index_type
        self.index_params = index_params or {}
        self.max_size = max_size
        self.eviction_policy = eviction_policy.lower()
        self.rebuild_threshold = rebuild_threshold

        self._index: Optional[BaseIndex] = None
        self._entries: list[CacheEntry] = []

        # Eviction bookkeeping
        self._active: set[int] = set()        # set of active cache_ids
        self._n_dead: int = 0                 # count of logically deleted entries
        self._access_order: OrderedDict = OrderedDict()  # LRU: id → None
        self._access_count: dict[int, int] = {}           # LFU: id → count
        self._n_evictions: int = 0

        # Statistics
        self._n_lookups = 0
        self._n_hits = 0
        self._n_inserts = 0
        self._n_rebuilds = 0
        self._total_lookup_time_ms = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, texts: list[str],
              responses: Optional[list[str]] = None,
              metadata_list: Optional[list[dict]] = None) -> float:
        """Build the cache from a set of embeddings and texts.

        If ``max_size`` is set and the input exceeds it, only the last
        ``max_size`` entries are retained (most recent first).

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

        # If bounded, keep only the last max_size entries
        if self.max_size > 0 and n > self.max_size:
            start = n - self.max_size
            embeddings = embeddings[start:]
            texts = texts[start:]
            responses = responses[start:]
            metadata_list = metadata_list[start:]
            n = self.max_size

        # Build entries
        self._entries = []
        self._active.clear()
        self._access_order.clear()
        self._access_count.clear()
        self._n_dead = 0

        for i in range(n):
            entry = CacheEntry(
                cache_id=i,
                query_text=texts[i],
                embedding=embeddings[i],
                response=responses[i],
                metadata=metadata_list[i],
            )
            self._entries.append(entry)
            self._active.add(i)
            self._access_order[i] = None
            self._access_count[i] = 0

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
            f"max_size={self.max_size}, "
            f"eviction={self.eviction_policy}, "
            f"build_time={build_time:.3f}s"
        )
        return build_time

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self, query_embedding: np.ndarray, k: int = 1,
        threshold: Optional[float] = None,
    ) -> LookupResult:
        """Search the cache for the nearest *active* entry.

        Args:
            query_embedding: Query vector, shape (D,) or (1, D).
            k: Number of nearest neighbors to return.
            threshold: Optional L2² distance threshold. When set, a
                neighbor is only counted as a "hit" if its distance
                is ≤ threshold.  When None, any valid neighbor is a hit.

        Returns:
            LookupResult with distances and nearest cache entry.
        """
        if self._index is None:
            raise ValueError("Cache not built. Call build() first.")

        # Reshape to (1, D) if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Fetch extra candidates to account for dead entries
        fetch_k = min(k + self._n_dead, len(self._entries))
        fetch_k = max(fetch_k, k)

        t_start = time.perf_counter()
        distances, indices = self._index.search(
            query_embedding.astype(np.float32), k=fetch_k
        )
        lookup_time_ms = (time.perf_counter() - t_start) * 1000

        # Filter to active entries only
        raw_dists = distances[0]
        raw_idxs = indices[0]
        active_dists = []
        active_idxs = []
        for d, idx in zip(raw_dists, raw_idxs):
            if int(idx) in self._active:
                active_dists.append(d)
                active_idxs.append(int(idx))
                if len(active_dists) >= k:
                    break

        # Pad if we didn't find enough active entries
        while len(active_dists) < k:
            active_dists.append(float("inf"))
            active_idxs.append(-1)

        active_dists = np.array(active_dists, dtype=np.float32)
        active_idxs = np.array(active_idxs, dtype=np.int64)

        # Get nearest active entry
        nearest_idx = active_idxs[0]
        nearest_dist = float(active_dists[0])

        if nearest_idx >= 0 and nearest_idx < len(self._entries):
            entry = self._entries[nearest_idx]
            if threshold is None or nearest_dist <= threshold:
                hit = True
            else:
                hit = False
        else:
            entry = None
            hit = False

        # Update eviction bookkeeping on hit
        if hit and nearest_idx >= 0:
            self._touch(nearest_idx)

        self._n_lookups += 1
        if hit:
            self._n_hits += 1
        self._total_lookup_time_ms += lookup_time_ms

        return LookupResult(
            hit=hit,
            distance=nearest_dist,
            cache_entry=entry,
            lookup_time_ms=lookup_time_ms,
            k_distances=active_dists,
            k_indices=active_idxs,
        )

    def batch_lookup(
        self, query_embeddings: np.ndarray, k: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch search — returns raw distances and indices.

        More efficient than calling lookup() in a loop because
        it uses a single FAISS search call.

        Note: This returns raw FAISS results and does NOT filter dead
        entries.  Use for evaluation where the caller handles filtering.

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

    # ------------------------------------------------------------------
    # Insert (with eviction)
    # ------------------------------------------------------------------

    def insert(
        self, embedding: np.ndarray, text: str,
        response: Optional[str] = None, metadata: Optional[dict] = None
    ):
        """Insert a new entry into the cache, evicting if necessary.

        Args:
            embedding: Embedding vector, shape (D,).
            text: Query text.
            response: Response text.
            metadata: Additional metadata.
        """
        if self._index is None:
            raise ValueError("Cache not built. Call build() first.")

        # Evict if at capacity
        if self.max_size > 0 and len(self._active) >= self.max_size:
            self._evict_one()

        cache_id = len(self._entries)
        entry = CacheEntry(
            cache_id=cache_id,
            query_text=text,
            embedding=embedding,
            response=response or text,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._active.add(cache_id)
        self._access_order[cache_id] = None
        self._access_count[cache_id] = 0

        # Add to FAISS index
        emb = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(emb)
        self._n_inserts += 1

        # Rebuild if too many dead entries
        self._maybe_rebuild()

    # ------------------------------------------------------------------
    # Eviction internals
    # ------------------------------------------------------------------

    def _touch(self, cache_id: int):
        """Record an access for eviction bookkeeping."""
        if self.eviction_policy == "lru":
            # Move to end (most recently used)
            self._access_order.move_to_end(cache_id)
        elif self.eviction_policy == "lfu":
            self._access_count[cache_id] = self._access_count.get(cache_id, 0) + 1

    def _evict_one(self):
        """Evict a single entry according to the eviction policy."""
        victim_id: Optional[int] = None

        if self.eviction_policy == "lru":
            # Evict least recently used (front of OrderedDict)
            for cid in self._access_order:
                if cid in self._active:
                    victim_id = cid
                    break
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            min_count = float("inf")
            for cid in self._active:
                cnt = self._access_count.get(cid, 0)
                if cnt < min_count:
                    min_count = cnt
                    victim_id = cid
        else:
            # No eviction policy — shouldn't reach here but be safe
            return

        if victim_id is not None:
            self._active.discard(victim_id)
            self._access_order.pop(victim_id, None)
            self._access_count.pop(victim_id, None)
            self._n_dead += 1
            self._n_evictions += 1

    def _maybe_rebuild(self):
        """Rebuild the FAISS index if dead entries exceed threshold."""
        total = len(self._entries)
        if total == 0 or self._n_dead / total < self.rebuild_threshold:
            return

        logger.info(
            f"Rebuilding index: {self._n_dead}/{total} dead entries "
            f"({self._n_dead/total:.1%} > {self.rebuild_threshold:.0%})"
        )

        # Collect active embeddings
        active_ids = sorted(self._active)
        active_embs = np.array(
            [self._entries[i].embedding for i in active_ids],
            dtype=np.float32,
        )

        # Rebuild mapping: old cache_id → new FAISS position
        id_remap = {old_id: new_pos for new_pos, old_id in enumerate(active_ids)}

        # Rebuild entries list (compacted)
        new_entries = []
        new_active = set()
        new_order = OrderedDict()
        new_count = {}
        for new_pos, old_id in enumerate(active_ids):
            old_entry = self._entries[old_id]
            new_entry = CacheEntry(
                cache_id=new_pos,
                query_text=old_entry.query_text,
                embedding=old_entry.embedding,
                response=old_entry.response,
                metadata=old_entry.metadata,
            )
            new_entries.append(new_entry)
            new_active.add(new_pos)
            new_order[new_pos] = None
            new_count[new_pos] = self._access_count.get(old_id, 0)

        self._entries = new_entries
        self._active = new_active
        self._access_order = new_order
        self._access_count = new_count
        self._n_dead = 0

        # Rebuild FAISS index
        self._index = create_index(
            self.index_type, dim=self.dim, **self.index_params
        )
        self._index.build(active_embs)
        self._n_rebuilds += 1

        logger.info(f"Rebuild complete: {len(new_entries)} active entries")

    # ------------------------------------------------------------------
    # Properties & stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of active entries in the cache."""
        return len(self._active)

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
            "total_entries": len(self._entries),
            "n_active": len(self._active),
            "n_dead": self._n_dead,
            "n_lookups": self._n_lookups,
            "n_hits": self._n_hits,
            "n_inserts": self._n_inserts,
            "n_evictions": self._n_evictions,
            "n_rebuilds": self._n_rebuilds,
            "index_type": self.index_type,
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy,
            "memory_mb": round(self.memory_mb, 2),
            "avg_lookup_time_ms": round(avg_lookup, 4),
        }
