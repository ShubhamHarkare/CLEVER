"""
Least-Recently-Used (LRU) eviction policy.

Evicts the entry that has not been accessed for the longest time.
Uses an ``OrderedDict`` where the *front* is the oldest access and
the *back* is the most recent.

This is the standard baseline eviction policy.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    """LRU eviction — evict the least recently used entry."""

    def __init__(self) -> None:
        # Front = least recently used, back = most recently used.
        self._order: OrderedDict[int, None] = OrderedDict()

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Move *cache_id* to the back (most recent)."""
        if cache_id in self._order:
            self._order.move_to_end(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """New entry starts at the back (most recent)."""
        self._order[cache_id] = None

    def on_evict(self, cache_id: int) -> None:
        """Remove evicted entry from the order."""
        self._order.pop(cache_id, None)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Return the front of the ordered dict (least recently used).

        Iterates from the front and returns the first ID that is still
        active.  In normal operation the front *is* active, but during
        edge cases (e.g. stale bookkeeping) we skip inactive IDs.
        """
        for cid in self._order:
            if cid in active_ids:
                return cid
        return None

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap all keys in the OrderedDict, preserving access order."""
        new_order: OrderedDict[int, None] = OrderedDict()
        for old_id in self._order:
            if old_id in id_remap:
                new_order[id_remap[old_id]] = None
        self._order = new_order

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lru"

    def __repr__(self) -> str:
        return f"LRUPolicy(tracked={len(self._order)})"
