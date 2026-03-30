"""
Index factory — create index instances from configuration.

Maps config strings/dicts to concrete BaseIndex subclasses.
Used by the benchmark runner to iterate over all configurations.
"""

import logging
from typing import Any

from src.indexes.base import BaseIndex
from src.indexes.flat_index import FlatIndex
from src.indexes.hnsw_index import HNSWIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.lsh_index import LSHIndex

logger = logging.getLogger(__name__)

INDEX_REGISTRY = {
    "flat": FlatIndex,
    "hnsw": HNSWIndex,
    "ivf": IVFIndex,
    "lsh": LSHIndex,
}


def create_index(index_type: str, dim: int, **params: Any) -> BaseIndex:
    """
    Create an index instance from a type string and parameters.

    Args:
        index_type: One of "flat", "hnsw", "ivf", "lsh".
        dim: Embedding dimensionality.
        **params: Index-specific parameters (e.g., M=32, efSearch=128).

    Returns:
        A BaseIndex instance (not yet built).

    Example:
        >>> idx = create_index("hnsw", dim=384, M=32, efConstruction=128, efSearch=64)
        >>> idx.build(vectors)
    """
    index_type = index_type.lower().strip()

    if index_type not in INDEX_REGISTRY:
        raise ValueError(
            f"Unknown index type '{index_type}'. "
            f"Available: {list(INDEX_REGISTRY.keys())}"
        )

    cls = INDEX_REGISTRY[index_type]
    index = cls(dim=dim, **params)
    logger.debug(f"Created {index_type} index with params={params}")
    return index


def list_index_types() -> list[str]:
    """Return available index type names."""
    return list(INDEX_REGISTRY.keys())
