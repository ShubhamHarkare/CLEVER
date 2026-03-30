"""
Abstract base class for cache eviction policies.

All eviction strategies (LRU, LFU, Semantic, Oracle) implement this
interface.  The ``SemanticCache`` delegates eviction decisions to a
policy instance, keeping FAISS/index logic separate from eviction logic.

Lifecycle hooks
---------------
- ``on_insert``  — called after a new entry is added to the cache.
- ``on_access``  — called on a cache hit (entry was read).
- ``select_victim`` — choose the entry to evict.
- ``on_evict``   — called after an entry has been logically deleted.
- ``on_rebuild`` — called after cache compaction remaps IDs.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EvictionPolicy(ABC):
    """Interface for cache eviction strategies.

    Invariants
    ----------
    1. ``select_victim`` is **only** called when the cache is at capacity
       and ``active_ids`` is non-empty.
    2. ``on_rebuild`` is called whenever the FAISS index is rebuilt and
       cache IDs are compacted.  Implementations must remap all internal
       state (OrderedDicts, counters, etc.).
    3. Policies must never mutate the ``active_ids`` set directly — they
       merely *advise* the cache on which ID to evict.
    """

    @abstractmethod
    def on_access(self, cache_id: int) -> None:
        """Record a cache hit on *cache_id*."""

    @abstractmethod
    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Record that *cache_id* was just inserted.

        Args:
            cache_id: The ID assigned to the new entry.
            embedding: The embedding vector (shape ``(D,)``).  Policies
                that don't need embeddings may ignore this.
        """

    @abstractmethod
    def on_evict(self, cache_id: int) -> None:
        """Clean up internal state after *cache_id* was evicted."""

    @abstractmethod
    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Choose the entry to evict.

        Args:
            active_ids: Set of currently active cache IDs.

        Returns:
            The cache_id to evict, or ``None`` if eviction is not
            possible (e.g. empty set — should not happen in practice).
        """

    @abstractmethod
    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap internal state after cache compaction.

        During a rebuild the cache re-numbers all active entries from 0.
        ``id_remap`` maps ``old_cache_id → new_cache_id`` for every
        active entry.  Evicted / dead IDs are not present in the map.

        Args:
            id_remap: Mapping from old IDs to new (compacted) IDs.
        """

    @property
    def name(self) -> str:
        """Human-readable policy name (e.g. ``'lru'``, ``'semantic'``)."""
        return self.__class__.__name__.lower().replace("policy", "")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
