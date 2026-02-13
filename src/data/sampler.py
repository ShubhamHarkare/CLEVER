"""
Deterministic subset sampler for creating scale-point datasets.

Creates reproducible subsets at 10K, 50K, 100K, 500K sizes from
the full processed query set for benchmarking at different scales.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard scale points
DEFAULT_SUBSET_SIZES = {
    "10k": 10_000,
    "50k": 50_000,
    "100k": 100_000,
    "500k": 500_000,
}


def create_subsets(
    df: pd.DataFrame,
    output_dir: str | Path,
    sizes: Optional[dict[str, int]] = None,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Create deterministic subsets of the query DataFrame.

    Each subset is saved as a parquet file. If the full dataset is smaller
    than a requested size, that subset is skipped with a warning.

    Args:
        df: Full processed queries DataFrame.
        output_dir: Directory to save subset parquet files.
        sizes: Dict mapping name → count. Defaults to DEFAULT_SUBSET_SIZES.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping subset name → saved file path.
    """
    if sizes is None:
        sizes = DEFAULT_SUBSET_SIZES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    saved = {}

    for name, size in sorted(sizes.items(), key=lambda x: x[1]):
        if size > len(df):
            logger.warning(
                f"Subset '{name}' ({size}) exceeds dataset size ({len(df)}), skipping"
            )
            continue

        indices = rng.choice(len(df), size=size, replace=False)
        indices.sort()  # Keep original ordering
        subset = df.iloc[indices].reset_index(drop=True)
        subset["query_id"] = subset.index

        path = output_dir / f"{name}_queries.parquet"
        subset.to_parquet(path, index=False)
        saved[name] = path
        logger.info(f"Created subset '{name}': {len(subset)} queries → {path}")

    # Also save the full dataset
    full_path = output_dir / "full_queries.parquet"
    df.to_parquet(full_path, index=False)
    saved["full"] = full_path
    logger.info(f"Saved full dataset: {len(df)} queries → {full_path}")

    return saved


def create_embedding_subsets(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    output_dir: str | Path,
    sizes: Optional[dict[str, int]] = None,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Create deterministic subsets of embeddings matching the query subsets.

    Uses the same random seed as create_subsets() so indices align.

    Args:
        embeddings: Full embedding array (N, D).
        df: Full processed queries DataFrame (for size reference).
        output_dir: Directory to save .npy embedding files.
        sizes: Dict mapping name → count. Defaults to DEFAULT_SUBSET_SIZES.
        seed: Random seed (must match create_subsets).

    Returns:
        Dict mapping subset name → saved .npy path.
    """
    if sizes is None:
        sizes = DEFAULT_SUBSET_SIZES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    saved = {}

    for name, size in sorted(sizes.items(), key=lambda x: x[1]):
        if size > len(df):
            logger.warning(
                f"Embedding subset '{name}' ({size}) exceeds dataset size ({len(df)}), skipping"
            )
            continue

        indices = rng.choice(len(df), size=size, replace=False)
        indices.sort()
        subset_emb = embeddings[indices]

        path = output_dir / f"{name}_embeddings.npy"
        np.save(path, subset_emb)
        saved[name] = path
        logger.info(f"Created embedding subset '{name}': {subset_emb.shape} → {path}")

    # Save full embeddings
    full_path = output_dir / "full_embeddings.npy"
    np.save(full_path, embeddings)
    saved["full"] = full_path
    logger.info(f"Saved full embeddings: {embeddings.shape} → {full_path}")

    return saved
