"""
Flat (brute-force) index wrapper — exact nearest-neighbor search.

Serves as the ground truth baseline for recall@k computation.
Uses IndexIDMap wrapping IndexFlatL2 to support removal by ID.
"""

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from src.indexes.base import BaseIndex

logger = logging.getLogger(__name__)


class FlatIndex(BaseIndex):
    """Brute-force exact search using FAISS IndexFlatL2."""

    def __init__(self, dim: int, metric: str = "L2"):
        """
        Args:
            dim: Embedding dimensionality (e.g. 384).
            metric: Distance metric — "L2" or "IP" (inner product).
        """
        self.dim = dim
        self.metric = metric
        self._params = {"metric": metric}

        if metric == "IP":
            base = faiss.IndexFlatIP(dim)
        else:
            base = faiss.IndexFlatL2(dim)

        # Wrap with IDMap so we can support removal
        self._index = faiss.IndexIDMap(base)
        self._next_id = 0

    def build(self, vectors: np.ndarray) -> None:
        """Build index from vectors. Assigns sequential IDs starting at 0."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        ids = np.arange(n, dtype=np.int64)
        self._index.add_with_ids(vectors, ids)
        self._next_id = n
        logger.info(f"FlatIndex built: {n} vectors, dim={self.dim}, metric={self.metric}")

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors. Returns (distances, indices)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors with auto-incrementing IDs."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._index.add_with_ids(vectors, ids)
        self._next_id += n

    def remove(self, ids: np.ndarray) -> None:
        """Remove vectors by their IDs."""
        ids = np.asarray(ids, dtype=np.int64)
        self._index.remove_ids(ids)

    @property
    def memory_usage_bytes(self) -> int:
        """Estimated memory: N * D * 4 bytes (float32) + ID map overhead."""
        n = self._index.ntotal
        # Float32 vectors + int64 IDs
        return n * self.dim * 4 + n * 8

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: str) -> None:
        self._index = faiss.read_index(str(path))

    def __repr__(self) -> str:
        return f"FlatIndex(dim={self.dim}, metric={self.metric}, ntotal={self.ntotal})"
