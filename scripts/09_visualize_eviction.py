#!/usr/bin/env python3
"""
Phase 4 — Eviction policy visualization.

Generates paper-ready figures from eviction experiment results:

Existing:
  28 — Hit rate over time (cumulative, line per policy, subplot per cache size)
  29 — Final hit rate comparison (grouped bar chart)
  30 — Semantic coverage comparison (grouped bar, lower = better)
  31 — Eviction overhead comparison (bar, avg query time)

New:
  32 — Rolling hit rate over time (local behavior via sliding window)
  33 — Policy comparison heatmap (policy x cache_size matrix)
  34 — Hit rate vs overhead tradeoff (scatter + Pareto frontier)
  35 — Eviction frequency over time (cumulative eviction count)

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


def _save_fig(fig, output_dir: Path, name: str):
    """Save figure as PNG and PDF."""
    out_path = output_dir / f"{name}.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


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

    # Matrix completeness validation
    logger.info("Validating eviction results matrix completeness...")
    config = data.get("config", {})
    per_seed = data.get("per_seed", {})

    policies = config.get("policies", [])
    cache_sizes = config.get("cache_sizes_pct", [])
    seeds = config.get("seeds", [])

    if policies and cache_sizes and seeds:
        missing_cells = []
        for policy in policies:
            if policy not in per_seed:
                missing_cells.append(f"[{policy}] missing entirely")
                continue
            for cache_pct in cache_sizes:
                pct_key = f"{cache_pct:.2f}"
                if pct_key not in per_seed[policy]:
                    missing_cells.append(f"[{policy}][{pct_key}] missing entirely")
                    continue
                seed_runs = per_seed[policy][pct_key]
                if len(seed_runs) < len(seeds):
                    missing_cells.append(
                        f"[{policy}][{pct_key}] has {len(seed_runs)} seeds "
                        f"(expected {len(seeds)})"
                    )

        if missing_cells:
            logger.error("FATAL: Incomplete eviction experiment matrix detected.")
            for mc in missing_cells:
                logger.error(f"  - {mc}")
            logger.error("Cannot generate valid publication figures with incomplete evidence.")
            sys.exit(1)

    logger.info(f"Loaded and validated complete results from {fpath}")
    return data


def _extract_timeseries(per_seed: dict, policy_name: str, pct_key: str, field: str):
    """Extract a timeseries field from per-seed results.

    Returns:
        xs: List of query indices (from first seed).
        ys_mean: Mean across seeds at each point.
        ys_std: Std across seeds at each point.
        n_seeds: Number of seeds.
    """
    seed_runs = per_seed.get(policy_name, {}).get(pct_key, [])
    if not seed_runs:
        return None, None, None, 0

    all_x = []
    all_y = []
    for run in seed_runs:
        rates = run["cumulative_hit_rates"]
        all_x.append([r["query_idx"] for r in rates])
        all_y.append([r[field] for r in rates])

    xs = all_x[0]
    ys_mean = np.mean(all_y, axis=0)
    ys_std = np.std(all_y, axis=0)
    return xs, ys_mean, ys_std, len(seed_runs)


# ═════════════════════════════════════════════════════════════════
# Shared plot helpers
# ═════════════════════════════════════════════════════════════════

def _plot_timeseries_by_cache_size(
    per_seed: dict,
    config: dict,
    output_dir: Path,
    field: str,
    ylabel: str,
    suptitle: str,
    filename: str,
    percent_yaxis: bool = True,
):
    """Parameterized timeseries plot: subplot per cache size, line per policy."""
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
            xs, ys_mean, ys_std, n_seeds = _extract_timeseries(
                per_seed, policy_name, pct_key, field
            )
            if xs is None:
                continue

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
            if n_seeds > 1:
                ax.fill_between(
                    xs, ys_mean - ys_std, ys_mean + ys_std,
                    alpha=0.15, color=color,
                )

        ax.set_xlabel("Queries Processed")
        ax.set_title(f"Cache Size = {cache_pct:.0%}")
        if percent_yaxis:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel(ylabel)
    legend_loc = "lower right" if percent_yaxis else "upper left"
    axes[-1].legend(loc=legend_loc, framealpha=0.9)

    fig.suptitle(suptitle, fontsize=15, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


def _plot_grouped_bar(
    agg: dict,
    config: dict,
    output_dir: Path,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    filename: str,
    percent_yaxis: bool = False,
    legend_loc: str = "upper left",
):
    """Parameterized grouped bar chart: policies grouped by cache size."""
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
            means.append(a.get(mean_key, 0))
            stds.append(a.get(std_key, 0))

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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (n_policies - 1) / 2)
    ax.set_xticklabels([f"{p:.0%}" for p in cache_sizes])
    if percent_yaxis:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc=legend_loc, framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


# ═════════════════════════════════════════════════════════════════
# Figure 28: Hit Rate Over Time (Cumulative)
# ═════════════════════════════════════════════════════════════════

def plot_hit_rate_over_time(results: dict, output_dir: Path):
    """Line plot: cumulative hit rate vs queries processed."""
    _plot_timeseries_by_cache_size(
        results["per_seed"], results["config"], output_dir,
        field="cumulative_hit_rate",
        ylabel="Cumulative Hit Rate",
        suptitle="Eviction Policy: Hit Rate Over Time",
        filename="28_eviction_hit_rate_over_time",
    )


# ═════════════════════════════════════════════════════════════════
# Figure 29: Final Hit Rate Comparison (Grouped Bar)
# ═════════════════════════════════════════════════════════════════

def plot_final_hit_rate_comparison(results: dict, output_dir: Path):
    """Grouped bar chart: final hit rate by policy and cache size."""
    _plot_grouped_bar(
        results["aggregated"], results["config"], output_dir,
        mean_key="hit_rate_mean", std_key="hit_rate_std",
        ylabel="Final Hit Rate",
        title="Eviction Policy Comparison: Final Hit Rate",
        filename="29_eviction_final_hit_rate",
        percent_yaxis=True,
    )


# ═════════════════════════════════════════════════════════════════
# Figure 30: Semantic Coverage Comparison
# ═════════════════════════════════════════════════════════════════

def plot_semantic_coverage(results: dict, output_dir: Path):
    """Bar chart: average min-distance from uncached queries to cache."""
    _plot_grouped_bar(
        results["aggregated"], results["config"], output_dir,
        mean_key="coverage_mean", std_key="coverage_std",
        ylabel="Avg. L2² Distance to Nearest Cached Entry",
        title="Semantic Coverage Comparison (Lower = Better)",
        filename="30_eviction_semantic_coverage",
        legend_loc="upper right",
    )


# ═════════════════════════════════════════════════════════════════
# Figure 31: Eviction Overhead Comparison
# ═════════════════════════════════════════════════════════════════

def plot_eviction_overhead(results: dict, output_dir: Path):
    """Bar chart: average query processing time per policy."""
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
    _save_fig(fig, output_dir, "31_eviction_overhead")


# ═════════════════════════════════════════════════════════════════
# Figure 32: Rolling Hit Rate Over Time
# ═════════════════════════════════════════════════════════════════

def plot_rolling_hit_rate(results: dict, output_dir: Path):
    """Line plot: rolling (windowed) hit rate vs queries processed."""
    _plot_timeseries_by_cache_size(
        results["per_seed"], results["config"], output_dir,
        field="rolling_hit_rate",
        ylabel="Rolling Hit Rate (window=1000)",
        suptitle="Eviction Policy: Rolling Hit Rate",
        filename="32_eviction_rolling_hit_rate",
    )


# ═════════════════════════════════════════════════════════════════
# Figure 33: Policy Comparison Heatmap
# ═════════════════════════════════════════════════════════════════

def plot_policy_heatmap(results: dict, output_dir: Path):
    """Heatmap: policy x cache_size -> mean hit rate.

    Annotated cells show hit rate +/- std.  Color scale highlights
    relative performance across the matrix.
    """
    agg = results["aggregated"]
    config = results["config"]
    policies = config["policies"]
    cache_sizes = config["cache_sizes_pct"]

    n_policies = len(policies)
    n_sizes = len(cache_sizes)

    # Build matrix
    matrix = np.zeros((n_policies, n_sizes))
    annotations = [[None] * n_sizes for _ in range(n_policies)]

    for i, policy_name in enumerate(policies):
        for j, cache_pct in enumerate(cache_sizes):
            pct_key = f"{cache_pct:.2f}"
            a = agg.get(policy_name, {}).get(pct_key, {})
            mean = a.get("hit_rate_mean", 0)
            std = a.get("hit_rate_std", 0)
            matrix[i, j] = mean
            annotations[i][j] = f"{mean:.3f}\n±{std:.3f}"

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_sizes), max(4, 1.2 * n_policies)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    # Annotate cells
    for i in range(n_policies):
        for j in range(n_sizes):
            text_color = "white" if matrix[i, j] > 0.5 * matrix.max() else "black"
            ax.text(
                j, i, annotations[i][j],
                ha="center", va="center", fontsize=10, color=text_color,
            )

    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f"{p:.0%}" for p in cache_sizes])
    ax.set_xlabel("Cache Size")

    ax.set_yticks(range(n_policies))
    ax.set_yticklabels([POLICY_LABELS.get(p, p) for p in policies])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Hit Rate")

    ax.set_title("Eviction Policy Comparison: Hit Rate Matrix")
    fig.tight_layout()
    _save_fig(fig, output_dir, "33_eviction_policy_heatmap")


# ═════════════════════════════════════════════════════════════════
# Figure 34: Hit Rate vs Overhead Tradeoff
# ═════════════════════════════════════════════════════════════════

def plot_hit_rate_vs_overhead(results: dict, output_dir: Path):
    """Scatter: hit rate (y) vs avg query time (x) per policy+cache_size.

    Draws a Pareto frontier — points where no other point has both
    higher hit rate and lower overhead.
    """
    agg = results["aggregated"]
    per_seed = results["per_seed"]
    config = results["config"]
    policies = config["policies"]
    cache_sizes = config["cache_sizes_pct"]

    fig, ax = plt.subplots(figsize=(8, 6))

    all_points = []  # (overhead, hit_rate) for Pareto

    for policy_name in policies:
        overheads = []
        hit_rates = []
        oh_stds = []
        hr_stds = []
        cache_labels = []

        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"

            # Hit rate from aggregated
            a = agg.get(policy_name, {}).get(pct_key, {})
            if not a:
                continue
            hr_mean = a["hit_rate_mean"]
            hr_std = a["hit_rate_std"]

            # Overhead from per-seed timing
            seed_runs = per_seed.get(policy_name, {}).get(pct_key, [])
            if not seed_runs:
                continue
            times = [r["timing"]["avg_query_time_ms"] for r in seed_runs]
            oh_mean = float(np.mean(times))
            oh_std = float(np.std(times))

            overheads.append(oh_mean)
            hit_rates.append(hr_mean)
            oh_stds.append(oh_std)
            hr_stds.append(hr_std)
            cache_labels.append(f"{cache_pct:.0%}")
            all_points.append((oh_mean, hr_mean))

        if not overheads:
            continue

        color = POLICY_COLORS.get(policy_name, "#333333")
        label = POLICY_LABELS.get(policy_name, policy_name)

        ax.errorbar(
            overheads, hit_rates,
            xerr=oh_stds, yerr=hr_stds,
            fmt="o", color=color, label=label,
            capsize=3, markersize=8, linewidth=1.5,
        )

        # Connect same-policy points (cache size increasing)
        if len(overheads) > 1:
            ax.plot(overheads, hit_rates, color=color, alpha=0.4, linewidth=1)

        # Annotate with cache size
        for oh, hr, cl in zip(overheads, hit_rates, cache_labels):
            ax.annotate(
                cl, (oh, hr),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, color=color, alpha=0.8,
            )

    # Draw Pareto frontier
    if all_points:
        pts = sorted(all_points, key=lambda p: p[0])
        pareto = [pts[0]]
        for p in pts[1:]:
            if p[1] >= pareto[-1][1]:
                pareto.append(p)
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.step(px, py, where="post", color="gray", linestyle=":", linewidth=1.5,
                    alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Avg. Query Time (ms)")
    ax.set_ylabel("Final Hit Rate")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title("Hit Rate vs. Overhead Tradeoff")
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, output_dir, "34_eviction_hit_rate_vs_overhead")


# ═════════════════════════════════════════════════════════════════
# Figure 35: Eviction Frequency Over Time
# ═════════════════════════════════════════════════════════════════

def plot_eviction_frequency(results: dict, output_dir: Path):
    """Line plot: cumulative eviction count vs queries processed."""
    _plot_timeseries_by_cache_size(
        results["per_seed"], results["config"], output_dir,
        field="n_evictions",
        ylabel="Cumulative Evictions",
        suptitle="Eviction Frequency Over Time",
        filename="35_eviction_frequency",
        percent_yaxis=False,
    )


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

    # Original figures (28-31)
    plot_hit_rate_over_time(results, output_dir)
    plot_final_hit_rate_comparison(results, output_dir)
    plot_semantic_coverage(results, output_dir)
    plot_eviction_overhead(results, output_dir)

    # New figures (32-35)
    plot_rolling_hit_rate(results, output_dir)
    plot_policy_heatmap(results, output_dir)
    plot_hit_rate_vs_overhead(results, output_dir)
    plot_eviction_frequency(results, output_dir)

    logger.info(f"\nAll 8 eviction figures saved to {output_dir}")


if __name__ == "__main__":
    main()
