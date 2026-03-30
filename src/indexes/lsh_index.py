"""
LSH (Locality-Sensitive Hashing) index wrapper.

Hashes vectors into binary codes; approximate search via Hamming distance.
Fastest build time but lowest recall. Useful as a baseline.
"""

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from src.indexes.base import BaseIndex

logger = logging.getLogger(__name__)


class LSHIndex(BaseIndex):
    """LSH-based ANN search using FAISS IndexLSH."""

    def __init__(self, dim: int, nbits: int = 768):
        """
        Args:
            dim: Embedding dimensionality.
            nbits: Number of hash bits. More bits = higher recall, more memory.
                Common choices: dim (384), 2*dim (768), 4*dim (1536).
        """
        self.dim = dim
        self.nbits = nbits
        self._params = {"nbits": nbits}
        self._index = faiss.IndexLSH(dim, nbits)

    def build(self, vectors: np.ndarray) -> None:
        """Build LSH index by hashing and adding all vectors."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        logger.info(f"LSHIndex built: {vectors.shape[0]} vectors, nbits={self.nbits}")

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors via Hamming distance."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to existing LSH index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def remove(self, ids: np.ndarray) -> None:
        """LSH does not support removal."""
        raise NotImplementedError(
            "LSH does not support vector removal. "
            "Rebuild the index without the unwanted vectors."
        )

    @property
    def memory_usage_bytes(self) -> int:
        """
        Estimated memory for LSH:
        - Binary codes: N * nbits / 8 bytes
        - Original vectors (if stored): N * D * 4 bytes
        - Hash functions: nbits * D * 4 bytes
        """
        n = self._index.ntotal
        code_bytes = n * self.nbits // 8
        hash_fn_bytes = self.nbits * self.dim * 4
        return code_bytes + hash_fn_bytes

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: str) -> None:
        self._index = faiss.read_index(str(path))

    def __repr__(self) -> str:
        return f"LSHIndex(dim={self.dim}, nbits={self.nbits}, ntotal={self.ntotal})"
