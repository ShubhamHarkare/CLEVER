"""
Semantic-aware eviction policy — the paper's novel contribution.

Scores each cached entry by a ratio of its *redundancy* (how many
similar entries are nearby) to its *utility* (recency + frequency).
Entries that are highly redundant and rarely used are evicted first;
entries that are isolated (no similar neighbours) or heavily used are
protected.

Eviction score
--------------
    score(e) = redundancy(e) / (α·recency(e) + β·frequency(e) + ε)

- ``redundancy(e)``: fraction of active cache entries whose L2²
  distance to *e* is ≤ ``similarity_threshold``.  High redundancy
  means evicting *e* won't reduce the cache's topical coverage.

- ``recency(e)``: normalised recency in [0, 1] where 1 = most
  recently accessed.  Recency is measured by rank in access order.

- ``frequency(e)``: normalised access count in [0, 1] where 1 = the
  most frequently accessed entry.

- ``α``, ``β``: tunable weights (default 1.0 each).

- ``ε``: small constant (1e-9) to avoid division by zero.

Batch optimisation
------------------
Computing neighbor counts for every entry on every eviction would be
O(N²) per eviction.  Instead we **batch-recompute** redundancy scores
every ``recompute_interval`` evictions using the FAISS index, and
cache the scores in between.  Between recomputations we update only
recency and frequency (which are cheap O(1) updates).
"""

import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy

logger = logging.getLogger(__name__)


class SemanticPolicy(EvictionPolicy):
    """Semantic-aware eviction policy.

    Args:
        similarity_threshold: L2² distance threshold for counting an
            entry as a "neighbour".  For normalised embeddings,
            ``cosine_sim ≈ 1 - L2²/2``, so ``L2² ≤ 0.30`` corresponds
            to ``cosine_sim ≥ 0.85``.
        alpha: Weight for the recency component.
        beta: Weight for the frequency component.
        recompute_interval: Number of evictions between full
            redundancy re-computations.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        alpha: float = 1.0,
        beta: float = 1.0,
        recompute_interval: int = 50,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.beta = beta
        self.recompute_interval = recompute_interval
        self._epsilon = 1e-9

        # ── Internal state ───────────────────────────────────────
        # Access order: front = oldest access, back = most recent.
        self._access_order: OrderedDict[int, None] = OrderedDict()
        # Access counts per entry.
        self._access_counts: dict[int, int] = {}
        # Cached redundancy scores (fraction of neighbours).
        self._redundancy: dict[int, float] = {}
        # Embeddings stored per cache_id (needed for neighbour counting).
        self._embeddings: dict[int, np.ndarray] = {}
        # Counter of evictions since last redundancy recomputation.
        self._evictions_since_recompute: int = 0

        # Timing
        self._total_eviction_time_s: float = 0.0
        self._n_evictions: int = 0
        self._n_recomputes: int = 0

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Move to back of access order and increment count."""
        if cache_id in self._access_order:
            self._access_order.move_to_end(cache_id)
        if cache_id in self._access_counts:
            self._access_counts[cache_id] += 1

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Register a newly inserted entry."""
        self._access_order[cache_id] = None
        self._access_counts[cache_id] = 0
        self._embeddings[cache_id] = embedding.copy()
        # New entries start with unknown redundancy — they'll be
        # scored on the next batch recomputation.  For now, assign 0
        # (isolated) so they are *not* immediately evicted.
        self._redundancy[cache_id] = 0.0

    def on_evict(self, cache_id: int) -> None:
        """Clean up all bookkeeping for the evicted entry."""
        self._access_order.pop(cache_id, None)
        self._access_counts.pop(cache_id, None)
        self._redundancy.pop(cache_id, None)
        self._embeddings.pop(cache_id, None)
        self._evictions_since_recompute += 1
        self._n_evictions += 1

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Select the entry with the highest eviction score."""
        if not active_ids:
            return None

        t_start = time.perf_counter()

        # Recompute redundancy scores periodically
        if self._evictions_since_recompute >= self.recompute_interval:
            self._recompute_redundancy(active_ids)
            self._evictions_since_recompute = 0

        # ── Compute per-entry scores ─────────────────────────────
        # Normalise recency: rank / N  (higher = more recent = safer)
        n_active = len(active_ids)
        access_list = [
            cid for cid in self._access_order if cid in active_ids
        ]
        recency = {}
        for rank, cid in enumerate(access_list):
            # rank 0 = oldest → recency 0;  rank N-1 = newest → recency 1
            recency[cid] = rank / max(n_active - 1, 1)

        # Normalise frequency: count / max_count
        counts = {
            cid: self._access_counts.get(cid, 0) for cid in active_ids
        }
        max_count = max(counts.values()) if counts else 1
        max_count = max(max_count, 1)  # avoid div-by-zero

        best_score = -1.0
        victim: Optional[int] = None

        for cid in active_ids:
            r = self._redundancy.get(cid, 0.0)
            rec = recency.get(cid, 0.0)
            freq = counts[cid] / max_count

            utility = self.alpha * rec + self.beta * freq + self._epsilon
            score = r / utility

            if score > best_score:
                best_score = score
                victim = cid

        self._total_eviction_time_s += time.perf_counter() - t_start
        return victim

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap all internal state after cache compaction."""
        # Access order
        new_order: OrderedDict[int, None] = OrderedDict()
        for old_id in self._access_order:
            if old_id in id_remap:
                new_order[id_remap[old_id]] = None
        self._access_order = new_order

        # Access counts
        new_counts: dict[int, int] = {}
        for old_id, cnt in self._access_counts.items():
            if old_id in id_remap:
                new_counts[id_remap[old_id]] = cnt
        self._access_counts = new_counts

        # Redundancy scores
        new_red: dict[int, float] = {}
        for old_id, score in self._redundancy.items():
            if old_id in id_remap:
                new_red[id_remap[old_id]] = score
        self._redundancy = new_red

        # Embeddings
        new_embs: dict[int, np.ndarray] = {}
        for old_id, emb in self._embeddings.items():
            if old_id in id_remap:
                new_embs[id_remap[old_id]] = emb
        self._embeddings = new_embs

        # Force a recomputation on the next eviction
        self._evictions_since_recompute = self.recompute_interval

    # ── Redundancy computation ───────────────────────────────────

    def _recompute_redundancy(self, active_ids: set[int]) -> None:
        """Batch-recompute redundancy scores for all active entries.

        Uses GPU acceleration via PyTorch if available, falling back to NumPy.
        """
        if not active_ids:
            return

        ids = sorted(active_ids)
        n = len(ids)

        embs = np.array(
            [self._embeddings[cid] for cid in ids],
            dtype=np.float32,
        )

        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            import torch
            if torch.cuda.is_available():
                self._recompute_redundancy_torch(ids, embs, n)
                return
        except ImportError:
            pass
            
        self._recompute_redundancy_numpy(ids, embs, n)

    def _recompute_redundancy_torch(self, ids: list[int], embs: np.ndarray, n: int) -> None:
        """GPU-accelerated redundancy computation using PyTorch."""
        import torch
        
        # Move everything to GPU
        embs_t = torch.from_numpy(embs).cuda()
        norms_sq = torch.sum(embs_t ** 2, dim=1)
        neighbour_counts = torch.zeros(n, dtype=torch.int32, device='cuda')
        
        # We can use a larger batch size on an RTX 6000 (e.g. 8192)
        batch_size = 8192
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_embs = embs_t[i:end]
            
            dot = torch.matmul(batch_embs, embs_t.T)
            # dist_sq = norms_sq_i + norms_sq_j - 2 * dot
            dist_sq = norms_sq[i:end].unsqueeze(1) + norms_sq.unsqueeze(0) - 2 * dot
            
            is_neighbour = dist_sq <= self.similarity_threshold
            
            # Exclude self explicitly
            for r in range(end - i):
                is_neighbour[r, i + r] = False
                
            neighbour_counts[i:end] = is_neighbour.sum(dim=1)
            
        counts_cpu = neighbour_counts.cpu().numpy()
        for i, cid in enumerate(ids):
            self._redundancy[cid] = float(counts_cpu[i]) / max(n - 1, 1)
            
        self._n_recomputes += 1

    def _recompute_redundancy_numpy(self, ids: list[int], embs: np.ndarray, n: int) -> None:
        """CPU fallback redundancy computation."""
        norms_sq = np.sum(embs ** 2, axis=1)  # (N,)
        neighbour_counts = np.zeros(n, dtype=np.int32)
        
        # Batch size for distance computation to prevent OOM
        # At B=2048, the dist_sq matrix is 2048 * N * 4 bytes
        # For N=60k, that's only ~490MB instead of ~14GB for N*N
        batch_size = 2048
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_embs = embs[i:end]
            
            # dot: (B, N)
            dot = batch_embs @ embs.T
            
            # dist_sq: (B, N)
            dist_sq = norms_sq[i:end, None] + norms_sq[None, :] - 2 * dot
            
            # Count neighbours within threshold
            is_neighbour = dist_sq <= self.similarity_threshold
            
            # Exclude self explicitly (diagonal elements)
            for r in range(end - i):
                is_neighbour[r, i + r] = False
                
            neighbour_counts[i:end] = is_neighbour.sum(axis=1)

        # Normalise: fraction of other active entries that are neighbours
        for i, cid in enumerate(ids):
            self._redundancy[cid] = neighbour_counts[i] / max(n - 1, 1)

        self._n_recomputes += 1

    # ── Stats ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return timing and scoring statistics."""
        avg_time = (
            self._total_eviction_time_s / self._n_evictions
            if self._n_evictions > 0 else 0.0
        )
        return {
            "n_evictions": self._n_evictions,
            "n_recomputes": self._n_recomputes,
            "avg_eviction_time_ms": round(avg_time * 1000, 4),
            "total_eviction_time_s": round(self._total_eviction_time_s, 3),
            "similarity_threshold": self.similarity_threshold,
            "recompute_interval": self.recompute_interval,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @property
    def name(self) -> str:
        return "semantic"

    def __repr__(self) -> str:
        return (
            f"SemanticPolicy(threshold={self.similarity_threshold}, "
            f"α={self.alpha}, β={self.beta}, "
            f"recompute_every={self.recompute_interval})"
        )
