#!/usr/bin/env python3
"""
03_run_index_benchmark.py — Phase 2: Compare FAISS index structures.

Runs Flat, HNSW, IVF, and LSH index benchmarks at specified scale,
measuring build time, memory, search latency, throughput, and recall@k.

Usage (local dev, 8.5K embeddings):
    python scripts/03_run_index_benchmark.py \
        --embed-dir results/embeddings/ \
        --size full \
        --output results/benchmarks/

Usage (Great Lakes, all scales):
    python scripts/03_run_index_benchmark.py \
        --embed-dir results/embeddings/ \
        --size 10k \
        --output results/benchmarks/ \
        --config configs/index_benchmark.yaml
"""

import json
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="configs/index_benchmark.yaml",
    help="Path to benchmark configuration YAML.",
)
@click.option(
    "--embed-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing embedding .npy files.",
)
@click.option(
    "--size",
    type=str,
    required=True,
    help="Scale label: '10k', '50k', '100k', '500k', or 'full'.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="results/benchmarks/",
    help="Output directory for JSON results.",
)
def main(config_path, embed_dir, size, output):
    """Run index benchmarks at a specified scale."""

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {config_path}")

    # Load embeddings
    embed_dir = Path(embed_dir)
    embed_path = embed_dir / f"{size}_embeddings.npy"

    if not embed_path.exists():
        logger.error(f"Embedding file not found: {embed_path}")
        sys.exit(1)

    logger.info(f"Loading embeddings from {embed_path}...")
    embeddings = np.load(embed_path)
    logger.info(f"Loaded: {embeddings.shape} ({embeddings.nbytes / 1e6:.1f} MB)")

    # Validate
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    dim = embeddings.shape[1]
    expected_dim = config.get("dataset", {}).get("embedding_dim", 384)
    assert dim == expected_dim, f"Dim mismatch: got {dim}, expected {expected_dim}"

    # Run benchmarks
    t0 = time.time()
    runner = BenchmarkRunner(
        embeddings=embeddings,
        config=config,
        output_dir=output,
        dataset_label=size,
    )
    results = runner.run()
    elapsed = time.time() - t0

    # Summary
    logger.info("=" * 70)
    logger.info(f"BENCHMARK COMPLETE — {size}")
    logger.info("=" * 70)
    logger.info(f"  Total configurations tested: {len(results)}")
    logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Brief summary table
    logger.info("")
    logger.info(f"{'Index':<10} {'Params':<35} {'Build(s)':<10} {'Mem(MB)':<10} "
                f"{'P50(ms)':<10} {'R@10':<8} {'QPS':<10}")
    logger.info("-" * 93)

    for r in results:
        if "error" in r:
            logger.info(f"{r['index_type']:<10} {str(r['params']):<35} ERROR: {r['error']}")
            continue

        params_str = str(r["params"])
        if len(params_str) > 33:
            params_str = params_str[:30] + "..."

        latency_p50 = r.get("search_latency_ms", {}).get("p50", -1)
        recall_10 = r.get("recall_at_10", -1)

        logger.info(
            f"{r['index_type']:<10} {params_str:<35} "
            f"{r['build_time_s']:<10.3f} {r['memory_mb']:<10.1f} "
            f"{latency_p50:<10.3f} {recall_10:<8.4f} "
            f"{r['throughput_qps']:<10.0f}"
        )

    output_path = Path(output) / f"index_benchmark_{size}.json"
    logger.info(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
