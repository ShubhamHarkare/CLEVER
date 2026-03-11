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

        # ── Pre-compute all future uses ──────────────────────────
        # dict mapping cache_id -> deque of stream indices where it will be accessed
        import collections
        self._future_uses: dict[int, collections.deque] = {}
        
        # next_use indicates the *immediate* next use for eviction sorting
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
        """Compute all future uses for each cache_id from stream_start onward.

        Uses batch NN search: for each stream query, find its nearest
        cache entry. Then, for each cache entry, append the stream position 
        to its queue of future uses.
        """
        import collections
        
        if len(cache_ids) == 0 or stream_start >= len(self._stream_embs):
            for cid in cache_ids:
                self._future_uses[cid] = collections.deque()
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

        # Initialise empty queues
        for cid in cache_ids:
            if cid not in self._future_uses:
                self._future_uses[cid] = collections.deque()

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

                # Append every future access directly to the queue (maintaining chronological order)
                if len(self._future_uses[cid]) == 0 or self._future_uses[cid][-1] < stream_idx:
                    self._future_uses[cid].append(stream_idx)
                    
        # Synchronize _next_use state
        for cid in cache_ids:
            self._update_next_use_from_queue(cid)
            
    def _update_next_use_from_queue(self, cid: int) -> None:
        """Helper to sync the _next_use float with the front of the queue."""
        q = self._future_uses.get(cid)
        if q and len(q) > 0:
            # Ensure the next use is actually in the future (>= stream_pos)
            while len(q) > 0 and q[0] < self._stream_pos:
                q.popleft()
                
            if len(q) > 0:
                self._next_use[cid] = float(q[0])
            else:
                self._next_use[cid] = _INF
        else:
            self._next_use[cid] = _INF

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """After an access, pop the current use and find the *next* time this entry is used."""
        q = self._future_uses.get(cache_id)
        if q and len(q) > 0:
            # The entry was just hit. It may have been the one at the front of the queue.
            # Pop anything past or present. 
            while len(q) > 0 and q[0] <= self._stream_pos:
                q.popleft()
        self._update_next_use_from_queue(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Track the new entry's all future uses in the remaining stream."""
        import collections
        
        self._cache_embs[cache_id] = embedding.copy()
        self._future_uses[cache_id] = collections.deque()

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
        for hit in hits:
            self._future_uses[cache_id].append(int(self._stream_pos + hit))
            
        self._update_next_use_from_queue(cache_id)

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
