#!/usr/bin/env python3
"""
Phase 4 — Eviction policy visualization.

Generates paper-ready figures from eviction experiment results:
  - Hit rate over time (line per policy, per cache size)
  - Final hit rate comparison (grouped bar chart)
  - Semantic coverage comparison
  - Eviction overhead comparison

Usage::

    python scripts/09_visualize_eviction.py \\
        --results results/eviction/eviction_results.json \\
        --output results/figures/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────
POLICY_COLORS = {
    "lru": "#4C72B0",
    "lfu": "#DD8452",
    "semantic": "#55A868",
    "oracle": "#C44E52",
}

POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "semantic": "Semantic (Ours)",
    "oracle": "Oracle (Upper Bound)",
}

POLICY_STYLES = {
    "lru": {"linestyle": "-", "marker": "o"},
    "lfu": {"linestyle": "-", "marker": "s"},
    "semantic": {"linestyle": "-", "marker": "D", "linewidth": 2.5},
    "oracle": {"linestyle": "--", "marker": "^"},
}


def setup_style():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ═════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════

def load_results(path: str) -> dict:
    """Load eviction results JSON."""
    fpath = Path(path)

    # Try multi-seed variant too
    if not fpath.exists():
        alt = fpath.parent / "eviction_results_multi_seed.json"
        if alt.exists():
            fpath = alt
            logger.info(f"Using multi-seed results: {fpath}")
        else:
            logger.error(f"Results not found: {path}")
            sys.exit(1)

    with open(fpath) as f:
        data = json.load(f)

    logger.info(f"Loaded results from {fpath}")
    return data


# ═════════════════════════════════════════════════════════════════
# Figure 28: Hit Rate Over Time
# ═════════════════════════════════════════════════════════════════

def plot_hit_rate_over_time(results: dict, output_dir: Path):
    """Line plot: cumulative hit rate vs queries processed.

    One subplot per cache size, one line per policy.
    """
    per_seed = results["per_seed"]
    config = results["config"]
    cache_sizes = config["cache_sizes_pct"]

    fig, axes = plt.subplots(
        1, len(cache_sizes),
        figsize=(6 * len(cache_sizes), 5),
        sharey=True,
    )
    if len(cache_sizes) == 1:
        axes = [axes]

    for ax_idx, cache_pct in enumerate(cache_sizes):
        ax = axes[ax_idx]
        pct_key = f"{cache_pct:.2f}"

        for policy_name in config["policies"]:
            if policy_name not in per_seed:
                continue
            if pct_key not in per_seed[policy_name]:
                continue

            seed_runs = per_seed[policy_name][pct_key]

            # Average cumulative_hit_rates across seeds
            all_x = []
            all_y = []
            for run in seed_runs:
                rates = run["cumulative_hit_rates"]
                xs = [r["query_idx"] for r in rates]
                ys = [r["cumulative_hit_rate"] for r in rates]
                all_x.append(xs)
                all_y.append(ys)

            # Find common x-axis (they should be the same)
            xs = all_x[0]
            ys_mean = np.mean(all_y, axis=0)
            ys_std = np.std(all_y, axis=0)

            style = POLICY_STYLES.get(policy_name, {})
            color = POLICY_COLORS.get(policy_name, "#333333")
            label = POLICY_LABELS.get(policy_name, policy_name)

            ax.plot(
                xs, ys_mean, color=color, label=label,
                linewidth=style.get("linewidth", 1.8),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", None),
                markevery=max(len(xs) // 6, 1),
                markersize=6,
            )
            if len(seed_runs) > 1:
                ax.fill_between(
                    xs, ys_mean - ys_std, ys_mean + ys_std,
                    alpha=0.15, color=color,
                )

        ax.set_xlabel("Queries Processed")
        ax.set_title(f"Cache Size = {cache_pct:.0%}")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel("Cumulative Hit Rate")
    axes[-1].legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Eviction Policy: Hit Rate Over Time", fontsize=15, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "28_eviction_hit_rate_over_time.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════
# Figure 29: Final Hit Rate Comparison (Grouped Bar)
# ═════════════════════════════════════════════════════════════════

def plot_final_hit_rate_comparison(results: dict, output_dir: Path):
    """Grouped bar chart: final hit rate by policy and cache size."""
    agg = results["aggregated"]
    config = results["config"]
    policies = config["policies"]
    cache_sizes = config["cache_sizes_pct"]

    n_policies = len(policies)
    n_sizes = len(cache_sizes)
    bar_width = 0.8 / n_policies
    x = np.arange(n_sizes)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, policy_name in enumerate(policies):
        means = []
        stds = []
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            a = agg.get(policy_name, {}).get(pct_key, {})
            means.append(a.get("hit_rate_mean", 0))
            stds.append(a.get("hit_rate_std", 0))

        color = POLICY_COLORS.get(policy_name, "#333333")
        label = POLICY_LABELS.get(policy_name, policy_name)

        ax.bar(
            x + i * bar_width,
            means, bar_width,
            yerr=stds, capsize=3,
            color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Cache Size (% of total queries)")
    ax.set_ylabel("Final Hit Rate")
    ax.set_title("Eviction Policy Comparison: Final Hit Rate")
    ax.set_xticks(x + bar_width * (n_policies - 1) / 2)
    ax.set_xticklabels([f"{p:.0%}" for p in cache_sizes])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_path = output_dir / "29_eviction_final_hit_rate.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════
# Figure 30: Semantic Coverage Comparison
# ═════════════════════════════════════════════════════════════════

def plot_semantic_coverage(results: dict, output_dir: Path):
    """Bar chart: average min-distance from uncached queries to cache.

    Lower distance = better coverage (cache covers query space well).
    """
    agg = results["aggregated"]
    config = results["config"]
    policies = config["policies"]
    cache_sizes = config["cache_sizes_pct"]

    n_policies = len(policies)
    n_sizes = len(cache_sizes)
    bar_width = 0.8 / n_policies
    x = np.arange(n_sizes)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, policy_name in enumerate(policies):
        means = []
        stds = []
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            a = agg.get(policy_name, {}).get(pct_key, {})
            means.append(a.get("coverage_mean", 0))
            stds.append(a.get("coverage_std", 0))

        color = POLICY_COLORS.get(policy_name, "#333333")
        label = POLICY_LABELS.get(policy_name, policy_name)

        ax.bar(
            x + i * bar_width,
            means, bar_width,
            yerr=stds, capsize=3,
            color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Cache Size (% of total queries)")
    ax.set_ylabel("Avg. L2² Distance to Nearest Cached Entry")
    ax.set_title("Semantic Coverage Comparison (Lower = Better)")
    ax.set_xticks(x + bar_width * (n_policies - 1) / 2)
    ax.set_xticklabels([f"{p:.0%}" for p in cache_sizes])
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = output_dir / "30_eviction_semantic_coverage.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════
# Figure 31: Eviction Overhead Comparison
# ═════════════════════════════════════════════════════════════════

def plot_eviction_overhead(results: dict, output_dir: Path):
    """Bar chart: average query processing time per policy.

    Shows trade-off: semantic policy has higher overhead but better
    hit rate.
    """
    per_seed = results["per_seed"]
    config = results["config"]
    policies = config["policies"]

    # Use the middle cache size for the comparison
    cache_sizes = config["cache_sizes_pct"]
    target_pct = cache_sizes[len(cache_sizes) // 2]
    pct_key = f"{target_pct:.2f}"

    fig, ax = plt.subplots(figsize=(7, 5))

    policy_times = {}
    for policy_name in policies:
        if policy_name not in per_seed:
            continue
        if pct_key not in per_seed[policy_name]:
            continue

        seed_runs = per_seed[policy_name][pct_key]
        times = [r["timing"]["avg_query_time_ms"] for r in seed_runs]
        policy_times[policy_name] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
        }

    x = np.arange(len(policy_times))
    names = list(policy_times.keys())
    means = [policy_times[n]["mean"] for n in names]
    stds = [policy_times[n]["std"] for n in names]
    colors = [POLICY_COLORS.get(n, "#333333") for n in names]
    labels = [POLICY_LABELS.get(n, n) for n in names]

    ax.bar(
        x, means, yerr=stds, capsize=4,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Avg. Query Time (ms)")
    ax.set_title(f"Eviction Overhead (Cache Size = {target_pct:.0%})")

    fig.tight_layout()
    out_path = output_dir / "31_eviction_overhead.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Eviction policy visualization"
    )
    parser.add_argument(
        "--results", default="results/eviction/eviction_results.json",
        help="Path to eviction results JSON",
    )
    parser.add_argument(
        "--output", default="results/figures/",
        help="Output directory for figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    results = load_results(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating eviction figures...")

    plot_hit_rate_over_time(results, output_dir)
    plot_final_hit_rate_comparison(results, output_dir)
    plot_semantic_coverage(results, output_dir)
    plot_eviction_overhead(results, output_dir)

    logger.info(f"\nAll eviction figures saved to {output_dir}")


if __name__ == "__main__":
    main()
