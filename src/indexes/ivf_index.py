"""
IVF (Inverted File) index wrapper — partition-based ANN search.

Clusters vectors into Voronoi cells; at search time, only scans
nearby cells (controlled by nprobe). Best for large-scale with
memory constraints when combined with PQ.
"""

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from src.indexes.base import BaseIndex

logger = logging.getLogger(__name__)


class IVFIndex(BaseIndex):
    """IVF partition-based ANN search using FAISS IndexIVFFlat."""

    def __init__(self, dim: int, nlist: int = 256, nprobe: int = 16):
        """
        Args:
            dim: Embedding dimensionality.
            nlist: Number of Voronoi cells (partitions).
                More cells = finer partitions, faster search, needs more training data.
            nprobe: Number of cells to scan at search time.
                Higher nprobe = better recall, slower search.
        """
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._params = {"nlist": nlist, "nprobe": nprobe}
        self._index = None  # Created during build (needs training)
        self._trained = False

    def build(self, vectors: np.ndarray) -> None:
        """
        Build IVF index: train centroids, then add vectors.

        If nlist > N, automatically adjusts nlist to N // 10 to avoid
        training failures.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]

        # Auto-adjust nlist if too large for dataset
        actual_nlist = self.nlist
        if actual_nlist > n:
            actual_nlist = max(1, n // 10)
            logger.warning(
                f"nlist={self.nlist} > N={n}, adjusting to nlist={actual_nlist}"
            )

        # Create quantizer and IVF index
        quantizer = faiss.IndexFlatL2(self.dim)
        self._index = faiss.IndexIVFFlat(quantizer, self.dim, actual_nlist)
        self._index.nprobe = self.nprobe

        # Train on the data
        self._index.train(vectors)
        self._trained = True

        # Add vectors
        self._index.add(vectors)
        logger.info(
            f"IVFIndex built: {n} vectors, nlist={actual_nlist}, "
            f"nprobe={self.nprobe}"
        )

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def set_nprobe(self, nprobe: int) -> None:
        """Change nprobe at query time (for parameter sweeps)."""
        self.nprobe = nprobe
        if self._index is not None:
            self._index.nprobe = nprobe

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to a trained IVF index."""
        if self._index is None or not self._trained:
            raise RuntimeError("Index not trained. Call build() first.")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def remove(self, ids: np.ndarray) -> None:
        """
        Remove vectors by their IDs.

        Note: FAISS IVFFlat supports removal via remove_ids(), but it is
        not efficient for large-scale use. Consider rebuilding instead.
        """
        if self._index is None:
            raise RuntimeError("Index not built.")
        ids_to_remove = np.asarray(ids, dtype=np.int64)
        selector = faiss.IDSelectorArray(len(ids_to_remove), faiss.swig_ptr(ids_to_remove))
        self._index.remove_ids(selector)

    @property
    def memory_usage_bytes(self) -> int:
        """
        Estimated memory for IVFFlat:
        - Vectors: N * D * 4 bytes (stored in inverted lists)
        - Centroids: nlist * D * 4 bytes
        - Inverted list metadata: nlist * 16 bytes
        """
        if self._index is None:
            return 0
        n = self._index.ntotal
        nlist = self._index.nlist
        vector_bytes = n * self.dim * 4
        centroid_bytes = nlist * self.dim * 4
        meta_bytes = nlist * 16
        return vector_bytes + centroid_bytes + meta_bytes

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def save(self, path: str) -> None:
        if self._index is None:
            raise RuntimeError("Index not built.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: str) -> None:
        self._index = faiss.read_index(str(path))
        self._trained = True

    def __repr__(self) -> str:
        nlist = self._index.nlist if self._index else self.nlist
        return (
            f"IVFIndex(dim={self.dim}, nlist={nlist}, "
            f"nprobe={self.nprobe}, ntotal={self.ntotal})"
        )
