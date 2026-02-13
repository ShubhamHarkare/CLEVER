"""
Data loader for LMSYS-Chat-1M dataset.

Downloads the dataset from HuggingFace and extracts the first user message
from each conversation for use as cached queries.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_lmsys_dataset(
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> "Dataset":
    """
    Download / load LMSYS-Chat-1M from HuggingFace.

    Args:
        cache_dir: Directory to cache the raw HuggingFace download.
        streaming: If True, stream data instead of downloading all at once.

    Returns:
        HuggingFace Dataset object.
    """
    logger.info("Loading LMSYS-Chat-1M dataset from HuggingFace...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )
    logger.info(f"Dataset loaded: {len(ds) if not streaming else '(streaming)'} conversations")
    return ds


def extract_first_user_queries(dataset, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Extract the first user message from each conversation.

    Each row in LMSYS-Chat-1M has a 'conversation' field which is a list
    of dicts like [{"role": "user", "content": "..."}, {"role": "assistant", ...}, ...].
    We extract only the first user turn — this represents the initial query
    that would be cached in a semantic cache system.

    Args:
        dataset: HuggingFace Dataset with 'conversation' column.
        max_rows: Optional limit on number of rows to process (for dev/debugging).

    Returns:
        DataFrame with columns: [query_id, query_text, original_index]
    """
    logger.info("Extracting first user queries from conversations...")

    records = []
    total = max_rows if max_rows else len(dataset)

    for idx, row in enumerate(tqdm(dataset, total=total, desc="Extracting queries")):
        if max_rows and idx >= max_rows:
            break

        conversation = row.get("conversation", [])
        if not conversation:
            continue

        # Find the first user message
        first_user_msg = None
        for turn in conversation:
            if turn.get("role") == "user":
                first_user_msg = turn.get("content", "").strip()
                break

        if first_user_msg:
            records.append({
                "query_id": idx,
                "query_text": first_user_msg,
                "original_index": idx,
            })

    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} user queries from {total} conversations")
    return df


def save_raw_queries(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save extracted queries to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} raw queries to {output_path}")
    return output_path


def load_processed_queries(path: str | Path) -> pd.DataFrame:
    """Load previously saved processed queries from parquet."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} queries from {path}")
    return df
