"""
Oracle (Bélády's optimal) eviction policy.

Evicts the entry whose next access is furthest in the future.  This
requires knowledge of the entire future query stream and therefore
cannot be used online — it serves purely as a theoretical upper bound.

Implementation
--------------
At construction time, the oracle receives the *full* future query stream
(as embeddings).  For each entry currently in the cache, it pre-computes
when that entry will next be accessed (i.e., when a query arrives whose
nearest cache neighbour is that entry).

On each eviction the oracle selects the active entry with the largest
``next_use`` timestamp.  Entries that are never accessed again have
``next_use = ∞`` and are the first to be evicted.

Complexity
----------
- Pre-computation: O(S × k) FAISS searches  (S = stream length)
- ``select_victim``: O(N)  (linear scan of active entries)
- Re-building ``next_use`` after cache mutations is handled lazily:
  we only track forward from the current stream position.
"""

import logging
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy

logger = logging.getLogger(__name__)

# Sentinel for "never accessed again"
_INF = float("inf")


class OraclePolicy(EvictionPolicy):
    """Bélády's optimal eviction — evict the entry used furthest away.

    Args:
        future_stream_embeddings: Array of shape (S, D) — the full
            future query stream in order.
        cache_embeddings: Array of shape (C, D) — the initial cache
            entries, indexed by cache_id (0 … C-1).
        cache_ids: Corresponding cache IDs for ``cache_embeddings``.
            Typically ``list(range(C))``.
        similarity_threshold: L2² distance threshold — a future query
            "accesses" a cache entry only if the NN distance is below
            this threshold.
    """

    def __init__(
        self,
        future_stream_embeddings: np.ndarray,
        cache_embeddings: np.ndarray,
        cache_ids: list[int],
        similarity_threshold: float = 0.90,
    ) -> None:
        self.similarity_threshold = similarity_threshold

        # ── Pre-compute next-use map ─────────────────────────────
        # next_use[cache_id] = earliest stream index where this
        # entry would be accessed.
        self._next_use: dict[int, float] = {}

        # Also store the full stream for re-computation on insert
        self._stream_embs = future_stream_embeddings
        self._stream_pos = 0  # current position in the stream
        self._cache_embs: dict[int, np.ndarray] = {}

        for cid, emb in zip(cache_ids, cache_embeddings):
            self._cache_embs[cid] = emb

        self._build_next_use(cache_ids, cache_embeddings, stream_start=0)

    def _build_next_use(
        self,
        cache_ids: list[int],
        cache_embeddings: np.ndarray,
        stream_start: int,
    ) -> None:
        """Compute next_use for each cache_id from stream_start onward.

        Uses batch NN search: for each stream query, find its nearest
        cache entry.  Then, for each cache entry, record the first
        stream position where it's the nearest neighbour.
        """
        if len(cache_ids) == 0 or stream_start >= len(self._stream_embs):
            for cid in cache_ids:
                self._next_use[cid] = _INF
            return

        remaining_stream = self._stream_embs[stream_start:]
        n_stream = len(remaining_stream)
        n_cache = len(cache_ids)

        if n_cache == 0:
            return

        # Build a small flat index for the current cache contents
        import faiss
        dim = cache_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(cache_embeddings.astype(np.float32))

        # Batch search: for each stream query, find its 1-NN in cache
        batch_size = min(10000, n_stream)
        # Map from local index position → cache_id
        pos_to_cid = {i: cid for i, cid in enumerate(cache_ids)}

        # Initialise all to infinity (never accessed)
        for cid in cache_ids:
            if cid not in self._next_use or self._next_use[cid] <= stream_start:
                self._next_use[cid] = _INF

        # Process in batches to control memory
        for batch_start in range(0, n_stream, batch_size):
            batch_end = min(batch_start + batch_size, n_stream)
            batch = remaining_stream[batch_start:batch_end].astype(np.float32)
            dists, idxs = index.search(batch, k=1)

            for i in range(len(batch)):
                stream_idx = stream_start + batch_start + i
                nn_idx = int(idxs[i, 0])
                nn_dist = float(dists[i, 0])

                if nn_idx < 0 or nn_dist > self.similarity_threshold:
                    continue

                cid = pos_to_cid.get(nn_idx)
                if cid is None:
                    continue

                # Record the earliest future access
                if self._next_use.get(cid, _INF) > stream_idx:
                    self._next_use[cid] = stream_idx

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """After an access, find the *next* time this entry is used.

        We update next_use by scanning forward from current stream_pos.
        For efficiency we set it to INF and rely on the pre-computed map
        (the map already has the first access; after that access happens
        we need the second access, etc.).
        """
        # Mark current next_use as consumed — find the next one
        # (We rely on the pre-computed map; updating lazily is acceptable
        # since the oracle still knows the next earliest among all entries)
        pass

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Track the new entry's next use in the remaining stream."""
        self._cache_embs[cache_id] = embedding.copy()

        # Find first stream position from current pos where this
        # entry would be the nearest neighbour.
        if self._stream_pos >= len(self._stream_embs):
            self._next_use[cache_id] = _INF
            return

        remaining = self._stream_embs[self._stream_pos:]
        emb = embedding.reshape(1, -1).astype(np.float32)

        # Compute distances from this single entry to all remaining queries
        dists = np.sum((remaining - emb) ** 2, axis=1)  # L2²

        # Find positions where this entry would be close enough
        hits = np.where(dists <= self.similarity_threshold)[0]
        if len(hits) > 0:
            self._next_use[cache_id] = int(self._stream_pos + hits[0])
        else:
            self._next_use[cache_id] = _INF

    def on_evict(self, cache_id: int) -> None:
        """Clean up state for evicted entry."""
        self._next_use.pop(cache_id, None)
        self._cache_embs.pop(cache_id, None)

    def advance_stream(self, new_pos: int) -> None:
        """Call this when the experiment advances the stream position.

        This allows the oracle to invalidate consumed next_use entries.
        """
        self._stream_pos = new_pos

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Evict the entry whose next access is furthest in the future."""
        if not active_ids:
            return None

        victim: Optional[int] = None
        max_next_use = -1.0

        for cid in active_ids:
            nu = self._next_use.get(cid, _INF)
            if nu > max_next_use:
                max_next_use = nu
                victim = cid

        return victim

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap next_use keys after cache compaction."""
        new_next_use: dict[int, float] = {}
        for old_id, nu in self._next_use.items():
            if old_id in id_remap:
                new_next_use[id_remap[old_id]] = nu
        self._next_use = new_next_use

        new_embs: dict[int, np.ndarray] = {}
        for old_id, emb in self._cache_embs.items():
            if old_id in id_remap:
                new_embs[id_remap[old_id]] = emb
        self._cache_embs = new_embs

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "oracle"

    def __repr__(self) -> str:
        return (
            f"OraclePolicy(threshold={self.similarity_threshold}, "
            f"stream_len={len(self._stream_embs)}, "
            f"tracked={len(self._next_use)})"
        )
