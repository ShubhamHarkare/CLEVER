#!/usr/bin/env python3
"""
02_generate_embeddings.py — Generate embeddings from processed queries.

Usage (local, small subsets):
    python scripts/02_generate_embeddings.py \
        --input data/processed_queries.parquet \
        --output results/embeddings/ \
        --device cpu --batch-size 64 \
        --sizes 10k,50k

Usage (Great Lakes, full dataset):
    python scripts/02_generate_embeddings.py \
        --input data/processed_queries.parquet \
        --output results/embeddings/ \
        --device cuda --batch-size 512 \
        --sizes 10k,50k,100k,500k,full
"""

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_processed_queries
from src.data.sampler import DEFAULT_SUBSET_SIZES, create_embedding_subsets
from src.embeddings.encoder import QueryEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to processed_queries.parquet.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="results/embeddings/",
    help="Output directory for embedding .npy files.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cpu",
    help="Device for encoding. Use 'cuda' on Great Lakes GPU nodes.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Encoding batch size. Use 512 for GPU, 64 for CPU.",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence-transformers model name.",
)
@click.option(
    "--sizes",
    default="10k,50k",
    help="Comma-separated subset sizes to generate (e.g., '10k,50k,100k,500k,full').",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Max queries to encode (for dev/debugging). None = all.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for subset creation.",
)
def main(input_path, output, device, batch_size, model, sizes, max_queries, seed):
    """Generate embeddings from processed queries and create scale subsets."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load processed queries
    logger.info("=" * 60)
    logger.info("Step 1: Loading processed queries")
    logger.info("=" * 60)
    df = load_processed_queries(input_path)

    if max_queries and max_queries < len(df):
        logger.info(f"Limiting to {max_queries} queries (dev mode)")
        df = df.head(max_queries).reset_index(drop=True)

    queries = df["query_text"].tolist()
    logger.info(f"Loaded {len(queries)} queries")

    # Step 2: Generate embeddings
    logger.info("=" * 60)
    logger.info(f"Step 2: Generating embeddings (device={device}, batch_size={batch_size})")
    logger.info("=" * 60)

    encoder = QueryEncoder(model_name=model, device=device)

    start_time = time.time()
    embeddings = encoder.encode(queries, batch_size=batch_size, normalize=True)
    elapsed = time.time() - start_time

    logger.info(f"Encoding complete in {elapsed:.1f}s ({len(queries)/elapsed:.0f} queries/sec)")

    # Validate embeddings
    logger.info("Validating embeddings...")
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
    assert embeddings.shape == (len(queries), encoder.embedding_dim)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings not properly L2-normalized"
    nan_count = np.isnan(embeddings).any(axis=1).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} embeddings with NaN — removing them")
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        embeddings = embeddings[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
    logger.info("✓ Embeddings validated (normalized, no NaN)")

    # Step 3: Create subsets
    logger.info("=" * 60)
    logger.info("Step 3: Creating embedding subsets")
    logger.info("=" * 60)

    # Parse requested sizes
    requested = [s.strip() for s in sizes.split(",")]
    subset_sizes = {}
    for name in requested:
        if name == "full":
            continue  # Full is always saved
        if name in DEFAULT_SUBSET_SIZES:
            subset_sizes[name] = DEFAULT_SUBSET_SIZES[name]
        else:
            logger.warning(f"Unknown subset size '{name}', skipping")

    saved = create_embedding_subsets(
        embeddings=embeddings,
        df=df,
        output_dir=output_dir,
        sizes=subset_sizes,
        seed=seed,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — Embedding Generation Summary")
    logger.info("=" * 60)
    logger.info(f"  Total queries encoded: {len(queries)}")
    logger.info(f"  Embedding dim:         {encoder.embedding_dim}")
    logger.info(f"  Encoding time:         {elapsed:.1f}s")
    logger.info(f"  Throughput:            {len(queries)/elapsed:.0f} queries/sec")
    logger.info(f"  Device:                {device}")
    logger.info(f"  Output directory:      {output_dir}")
    logger.info(f"  Subsets created:       {list(saved.keys())}")
    for name, path in saved.items():
        emb = np.load(path)
        size_mb = emb.nbytes / (1024 * 1024)
        logger.info(f"    {name}: {emb.shape} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
