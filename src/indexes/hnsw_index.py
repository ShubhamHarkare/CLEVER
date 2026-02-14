"""
HNSW index wrapper — Hierarchical Navigable Small World graph.

Best for low-latency, high-recall approximate nearest-neighbor search.
Tunable parameters: M (graph degree), efConstruction, efSearch.
"""

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from src.indexes.base import BaseIndex

logger = logging.getLogger(__name__)


class HNSWIndex(BaseIndex):
    """HNSW graph-based ANN search using FAISS IndexHNSWFlat."""

    def __init__(self, dim: int, M: int = 32, efConstruction: int = 128,
                 efSearch: int = 128):
        """
        Args:
            dim: Embedding dimensionality.
            M: Number of bi-directional links per node (graph degree).
                Higher M = better recall, more memory.
            efConstruction: Size of dynamic candidate list during build.
                Higher = better quality graph, slower build.
            efSearch: Size of candidate list during search.
                Higher = better recall, slower search.
        """
        self.dim = dim
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self._params = {"M": M, "efConstruction": efConstruction,
                        "efSearch": efSearch}

        self._index = faiss.IndexHNSWFlat(dim, M)
        self._index.hnsw.efConstruction = efConstruction
        self._index.hnsw.efSearch = efSearch

    def build(self, vectors: np.ndarray) -> None:
        """Build HNSW graph from vectors."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        logger.info(
            f"HNSWIndex built: {vectors.shape[0]} vectors, "
            f"M={self.M}, efConstruction={self.efConstruction}, "
            f"efSearch={self.efSearch}"
        )

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def set_ef_search(self, efSearch: int) -> None:
        """Change efSearch at query time (for parameter sweeps)."""
        self.efSearch = efSearch
        self._index.hnsw.efSearch = efSearch

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to existing HNSW graph."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def remove(self, ids: np.ndarray) -> None:
        """HNSW does not support removal."""
        raise NotImplementedError(
            "HNSW does not support vector removal. "
            "Rebuild the index without the unwanted vectors."
        )

    @property
    def memory_usage_bytes(self) -> int:
        """
        Estimated memory for HNSW:
        - Vectors: N * D * 4 bytes
        - Graph links: N * M * 2 * 4 bytes (bidirectional, 2 levels avg)
        - Overhead: ~10% for internal structures
        """
        n = self._index.ntotal
        vector_bytes = n * self.dim * 4
        graph_bytes = n * self.M * 2 * 4 * 2  # 2 levels on average
        overhead = int((vector_bytes + graph_bytes) * 0.1)
        return vector_bytes + graph_bytes + overhead

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: str) -> None:
        self._index = faiss.read_index(str(path))

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.dim}, M={self.M}, "
            f"efConstruction={self.efConstruction}, "
            f"efSearch={self.efSearch}, ntotal={self.ntotal})"
        )
