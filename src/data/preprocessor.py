"""
Query preprocessor for the semantic cache benchmark.

Filters, cleans, and deduplicates extracted user queries from LMSYS-Chat-1M.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def filter_queries(
    df: pd.DataFrame,
    min_tokens: int = 3,
    max_tokens: int = 512,
) -> pd.DataFrame:
    """
    Filter queries by length constraints.

    Removes:
      - Empty or whitespace-only queries
      - Queries shorter than min_tokens words
      - Queries longer than max_tokens words

    Args:
        df: DataFrame with 'query_text' column.
        min_tokens: Minimum word count (inclusive).
        max_tokens: Maximum word count (inclusive).

    Returns:
        Filtered DataFrame.
    """
    initial_count = len(df)

    # Remove empty / whitespace-only
    df = df[df["query_text"].str.strip().astype(bool)].copy()
    after_empty = len(df)
    logger.info(f"Removed {initial_count - after_empty} empty queries")

    # Compute token counts (simple whitespace split)
    df["token_count"] = df["query_text"].str.split().str.len()

    # Filter by length
    df = df[(df["token_count"] >= min_tokens) & (df["token_count"] <= max_tokens)].copy()
    after_length = len(df)
    logger.info(
        f"Removed {after_empty - after_length} queries outside "
        f"[{min_tokens}, {max_tokens}] token range"
    )

    logger.info(f"After filtering: {len(df)} queries (from {initial_count})")
    return df


def deduplicate_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate queries by exact text match.

    Keeps the first occurrence and adds a 'frequency' column recording
    how many times each query appeared.

    Args:
        df: DataFrame with 'query_text' column.

    Returns:
        Deduplicated DataFrame with added 'frequency' column.
    """
    initial_count = len(df)

    # Count duplicates
    freq = df.groupby("query_text").size().reset_index(name="frequency")

    # Keep first occurrence
    df_dedup = df.drop_duplicates(subset=["query_text"], keep="first").copy()

    # Merge frequency counts
    df_dedup = df_dedup.merge(freq, on="query_text", how="left")

    # Re-assign sequential query IDs
    df_dedup = df_dedup.reset_index(drop=True)
    df_dedup["query_id"] = df_dedup.index

    removed = initial_count - len(df_dedup)
    logger.info(f"Deduplicated: removed {removed} duplicates, {len(df_dedup)} unique queries remain")

    return df_dedup


def preprocess_queries(
    df: pd.DataFrame,
    min_tokens: int = 3,
    max_tokens: int = 512,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: filter + deduplicate.

    Args:
        df: Raw DataFrame with 'query_text' column.
        min_tokens: Minimum word count.
        max_tokens: Maximum word count.

    Returns:
        Cleaned, filtered, deduplicated DataFrame.
    """
    logger.info(f"Starting preprocessing of {len(df)} queries...")
    df = filter_queries(df, min_tokens=min_tokens, max_tokens=max_tokens)
    df = deduplicate_queries(df)
    logger.info(f"Preprocessing complete: {len(df)} queries remain")
    return df


def save_processed_queries(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save processed queries to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} processed queries to {output_path}")
    return output_path
