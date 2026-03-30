"""
Abstract base class for all FAISS index wrappers.

Every index implementation (Flat, HNSW, IVF, LSH) must subclass this
and implement all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseIndex(ABC):
    """Abstract interface for ANN index wrappers."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """
        Build the index from a set of vectors.

        For indexes that require training (e.g., IVF), this method
        handles both training and adding vectors.

        Args:
            vectors: Array of shape (N, D), dtype float32.
        """
        ...

    @abstractmethod
    def search(
        self, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors.

        Args:
            queries: Array of shape (Q, D), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices):
              - distances: shape (Q, k), L2 distances
              - indices: shape (Q, k), vector IDs
        """
        ...

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to an already-built index.

        Args:
            vectors: Array of shape (M, D), dtype float32.
        """
        ...

    @abstractmethod
    def remove(self, ids: np.ndarray) -> None:
        """
        Remove vectors by their IDs.

        Note: Not all FAISS index types support removal.
        Implementations should raise NotImplementedError if unsupported.

        Args:
            ids: Array of vector IDs to remove.
        """
        ...

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """Return estimated memory usage of the index in bytes."""
        ...

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Return total number of vectors in the index."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ntotal={self.ntotal})"
