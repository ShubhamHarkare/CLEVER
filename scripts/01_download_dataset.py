#!/usr/bin/env python3
"""
01_download_dataset.py — Download LMSYS-Chat-1M and preprocess queries.

Usage:
    python scripts/01_download_dataset.py --output data/
    python scripts/01_download_dataset.py --output data/ --max-rows 10000  # dev mode
"""

import logging
import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import extract_first_user_queries, load_lmsys_dataset, save_raw_queries
from src.data.preprocessor import preprocess_queries, save_processed_queries
from src.data.sampler import create_subsets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output",
    type=click.Path(),
    default="data/",
    help="Output directory for processed data.",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="HuggingFace cache directory for raw dataset download.",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Max rows to process (for development/debugging). None = all.",
)
@click.option(
    "--min-tokens",
    type=int,
    default=3,
    help="Minimum word count for query filtering.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum word count for query filtering.",
)
@click.option(
    "--skip-subsets",
    is_flag=True,
    default=False,
    help="Skip creating scale subsets (just process queries).",
)
def main(output, cache_dir, max_rows, min_tokens, max_tokens, skip_subsets):
    """Download LMSYS-Chat-1M, extract and preprocess user queries."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download dataset
    logger.info("=" * 60)
    logger.info("Step 1: Loading LMSYS-Chat-1M dataset")
    logger.info("=" * 60)
    dataset = load_lmsys_dataset(cache_dir=cache_dir)

    # Step 2: Extract first user queries
    logger.info("=" * 60)
    logger.info("Step 2: Extracting first user queries")
    logger.info("=" * 60)
    raw_df = extract_first_user_queries(dataset, max_rows=max_rows)
    raw_path = output_dir / "raw_queries.parquet"
    save_raw_queries(raw_df, raw_path)
    logger.info(f"Raw queries saved: {len(raw_df)} rows → {raw_path}")

    # Step 3: Preprocess (filter + deduplicate)
    logger.info("=" * 60)
    logger.info("Step 3: Preprocessing queries")
    logger.info("=" * 60)
    processed_df = preprocess_queries(
        raw_df, min_tokens=min_tokens, max_tokens=max_tokens
    )
    processed_path = output_dir / "processed_queries.parquet"
    save_processed_queries(processed_df, processed_path)

    # Step 4: Create scale subsets
    if not skip_subsets:
        logger.info("=" * 60)
        logger.info("Step 4: Creating scale subsets")
        logger.info("=" * 60)
        create_subsets(processed_df, output_dir=output_dir, seed=42)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — Data Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"  Raw queries:       {len(raw_df)}")
    logger.info(f"  After processing:  {len(processed_df)}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info(f"  Token count range: [{min_tokens}, {max_tokens}]")

    if "frequency" in processed_df.columns:
        dups = processed_df["frequency"].sum() - len(processed_df)
        logger.info(f"  Duplicates removed: {int(dups)}")

    logger.info("")
    logger.info("Next step: python scripts/02_generate_embeddings.py")


if __name__ == "__main__":
    main()
