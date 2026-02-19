#!/usr/bin/env python3
"""
Phase 3 — Routing Evaluation Visualizations.

Generates publication-quality plots from routing evaluation results:
  21. Threshold Sensitivity   — hit rate & cost savings vs threshold
  22. Cost Savings Breakdown  — stacked bar (cache vs LLM)
  23. Distance Distribution   — histogram of NN distances (hits vs misses)
  24. Quality–Hit Rate Curve  — ROC-style tradeoff
  25. Index Comparison        — routing metrics across Flat/HNSW/IVF
  26. Routing Summary         — key metrics table

Usage:
    python scripts/07_visualize_routing.py \\
        --results-dir results/routing/ \\
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
import matplotlib.ticker as mticker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── House style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#444466",
    "axes.labelcolor": "#ccccdd",
    "text.color": "#ccccdd",
    "xtick.color": "#aaaacc",
    "ytick.color": "#aaaacc",
    "grid.color": "#333355",
    "grid.alpha": 0.4,
    "legend.facecolor": "#2a2a3e",
    "legend.edgecolor": "#444466",
    "font.family": "sans-serif",
    "font.size": 12,
})

COLORS = {
    "hit_rate": "#06d6a0",
    "cost_savings": "#ffd166",
    "quality": "#ef476f",
    "latency": "#118ab2",
    "monetary": "#073b4c",
    "cache": "#06d6a0",
    "llm": "#ef476f",
    "flat": "#7b68ee",
    "hnsw": "#06d6a0",
    "ivf": "#ffa62f",
    "lsh": "#ef476f",
}

SAVE_DPI = 180


def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Load routing and comparison results."""
    routing_path = results_dir / "routing_eval.json"
    multi_seed_path = results_dir / "routing_eval_multi_seed.json"
    comparison_path = results_dir / "index_comparison.json"

    routing = {}
    comparison = {}

    if routing_path.exists():
        with open(routing_path) as f:
            routing = json.load(f)
        logger.info(f"Loaded routing results from {routing_path}")
    elif multi_seed_path.exists():
        # Fall back to multi-seed results: use the first per-seed run for plots
        with open(multi_seed_path) as f:
            multi = json.load(f)
        per_seed = multi.get("per_seed", {})
        if per_seed:
            first_seed = next(iter(per_seed))
            routing = per_seed[first_seed]
            logger.info(
                f"Loaded routing results from {multi_seed_path} "
                f"(seed={first_seed})"
            )
        else:
            logger.warning(f"No per-seed data in {multi_seed_path}")
    else:
        logger.warning(f"No routing results at {routing_path} or {multi_seed_path}")

    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)
        logger.info(f"Loaded comparison results from {comparison_path}")

    return routing, comparison


def _get_strategy_data(routing: dict, strategy: str = "random") -> dict:
    """Get data for the preferred strategy."""
    if strategy in routing:
        return routing[strategy]
    # Fallback to first non-meta strategy
    for k, v in routing.items():
        if k != "meta" and isinstance(v, dict):
            return v
    return {}


# ── Plot 21: Threshold Sensitivity ──────────────────────────────────

def plot_threshold_sensitivity(routing: dict, output_dir: Path):
    """Hit rate and cost savings vs. routing threshold."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    strategies = [k for k in routing if k != "meta"]

    for strategy in strategies:
        data = routing[strategy]
        sweep = data.get("threshold_sweep", [])
        if not sweep:
            continue

        thresholds = [r["threshold"] for r in sweep]
        hit_rates = [r["hit_rate"] for r in sweep]
        latency_savings = [r["latency_saving_pct"] for r in sweep]
        monetary_savings = [r["monetary_saving_pct"] for r in sweep]

        linestyle = "-" if strategy == "random" else "--"
        label_suffix = f" ({strategy})"

        ax1.plot(
            thresholds, [h * 100 for h in hit_rates],
            color=COLORS["hit_rate"], linewidth=2, markersize=5,
            marker="o", linestyle=linestyle,
            label=f"Hit Rate{label_suffix}", alpha=0.9,
        )

        ax2 = ax1.twinx()
        ax2.plot(
            thresholds, latency_savings,
            color=COLORS["cost_savings"], linewidth=2, markersize=5,
            marker="s", linestyle=linestyle,
            label=f"Latency Savings{label_suffix}", alpha=0.9,
        )
        ax2.plot(
            thresholds, monetary_savings,
            color=COLORS["latency"], linewidth=2, markersize=5,
            marker="^", linestyle=linestyle,
            label=f"Monetary Savings{label_suffix}", alpha=0.9,
        )

        # Mark adaptive threshold
        adaptive = data.get("adaptive", {})
        if adaptive and "best_threshold" in adaptive:
            ax1.axvline(
                adaptive["best_threshold"], color=COLORS["quality"],
                linestyle=":", linewidth=2, alpha=0.8,
                label=f"Adaptive θ={adaptive['best_threshold']:.2f}"
            )

    ax1.set_xlabel("Routing Threshold (L2² distance)")
    ax1.set_ylabel("Hit Rate (%)", color=COLORS["hit_rate"])
    ax1.tick_params(axis="y", labelcolor=COLORS["hit_rate"])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 105)

    ax2.set_ylabel("Savings (%)", color=COLORS["cost_savings"])
    ax2.tick_params(axis="y", labelcolor=COLORS["cost_savings"])
    ax2.set_ylim(0, 105)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    n_queries = routing.get("meta", {}).get("total_queries", "?")
    ax1.set_title(
        f"Threshold Sensitivity — {n_queries} queries",
        fontsize=15, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "21_threshold_sensitivity.png", dpi=SAVE_DPI)
    plt.close(fig)
    logger.info("✓ 21_threshold_sensitivity.png")


# ── Plot 22: Cost Savings Breakdown ─────────────────────────────────

def plot_cost_breakdown(routing: dict, output_dir: Path):
    """Stacked bar chart: cache hits vs LLM calls at key thresholds."""
    data = _get_strategy_data(routing)
    sweep = data.get("threshold_sweep", [])
    if not sweep:
        return

    # Select ~6 representative thresholds
    n = len(sweep)
    step = max(1, n // 6)
    selected = sweep[::step]
    if sweep[-1] not in selected:
        selected.append(sweep[-1])

    fig, ax = plt.subplots(figsize=(12, 6))

    thresholds = [f"θ={r['threshold']:.2f}" for r in selected]
    hits = [r["n_hits"] for r in selected]
    misses = [r["n_misses"] for r in selected]

    x = np.arange(len(thresholds))
    width = 0.6

    bars_cache = ax.bar(x, hits, width, label="Cache Hits",
                        color=COLORS["cache"], alpha=0.85)
    bars_llm = ax.bar(x, misses, width, bottom=hits, label="LLM Calls",
                      color=COLORS["llm"], alpha=0.85)

    # Add percentage labels on bars
    total = [h + m for h, m in zip(hits, misses)]
    for i, (h, t) in enumerate(zip(hits, total)):
        pct = h / t * 100 if t > 0 else 0
        ax.text(i, h / 2, f"{pct:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        if t - h > 0:
            ax.text(i, h + (t - h) / 2, f"{(t-h)/t*100:.0f}%",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, rotation=30, ha="right")
    ax.set_ylabel("Number of Queries")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    n_queries = routing.get("meta", {}).get("total_queries", "?")
    ax.set_title(
        f"Routing Breakdown — {n_queries} queries",
        fontsize=15, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "22_cost_breakdown.png", dpi=SAVE_DPI)
    plt.close(fig)
    logger.info("✓ 22_cost_breakdown.png")


# ── Plot 23: Distance Distribution ──────────────────────────────────

def plot_distance_distribution(routing: dict, output_dir: Path):
    """Histogram of nearest-neighbor distances."""
    data = _get_strategy_data(routing)
    dist_info = data.get("distance_analysis", {})
    histogram = dist_info.get("histogram", {})

    if not histogram:
        return

    bins = histogram.get("bins", [])
    counts = histogram.get("counts", [])

    if len(bins) < 2 or len(counts) < 1:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    bar_width = bins[1] - bins[0]

    ax.bar(
        bin_centers, counts, width=bar_width * 0.9,
        color=COLORS["hit_rate"], alpha=0.75, edgecolor="#333",
        label="All queries",
    )

    # Mark key thresholds
    adaptive = data.get("adaptive", {})
    if adaptive and "best_threshold" in adaptive:
        ax.axvline(
            adaptive["best_threshold"], color=COLORS["quality"],
            linestyle="--", linewidth=2,
            label=f"Adaptive θ={adaptive['best_threshold']:.2f}",
        )

    # Mark percentiles
    for pname, pval in [("P25", dist_info.get("p25")),
                        ("Median", dist_info.get("median")),
                        ("P75", dist_info.get("p75"))]:
        if pval is not None:
            ax.axvline(pval, color=COLORS["cost_savings"],
                       linestyle=":", linewidth=1.5, alpha=0.7)
            ax.text(pval, max(counts) * 0.95, f" {pname}={pval:.2f}",
                    fontsize=9, color=COLORS["cost_savings"],
                    rotation=90, va="top")

    ax.set_xlabel("L2² Distance to Nearest Cached Query")
    ax.set_ylabel("Number of Queries")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    n_queries = data.get("eval_size", "?")
    ax.set_title(
        f"Cache Distance Distribution — {n_queries} eval queries",
        fontsize=15, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "23_distance_distribution.png", dpi=SAVE_DPI)
    plt.close(fig)
    logger.info("✓ 23_distance_distribution.png")


# ── Plot 24: Quality vs Hit Rate (ROC-style) ────────────────────────

def plot_quality_hitrate(routing: dict, output_dir: Path):
    """ROC-style curve: quality vs hit rate across thresholds."""
    fig, ax = plt.subplots(figsize=(10, 8))

    strategies = [k for k in routing if k != "meta"]

    for strategy in strategies:
        data = routing[strategy]
        sweep = data.get("threshold_sweep", [])
        if not sweep:
            continue

        hit_rates = [r["hit_rate"] * 100 for r in sweep]
        qualities = [r.get("cosine_quality", r.get("quality", 1.0)) * 100 for r in sweep]

        marker = "o" if strategy == "random" else "s"
        ax.plot(
            hit_rates, qualities,
            f"{marker}-", linewidth=2, markersize=6,
            label=f"{strategy} fill", alpha=0.85,
        )

        # Annotate a few points with threshold values
        for i in range(0, len(sweep), max(1, len(sweep) // 5)):
            r = sweep[i]
            ax.annotate(
                f"θ={r['threshold']:.1f}",
                (r["hit_rate"] * 100, r.get("cosine_quality", r.get("quality", 1.0)) * 100),
                fontsize=8, alpha=0.7,
                xytext=(5, -10), textcoords="offset points",
            )

        # Mark adaptive point
        adaptive = data.get("adaptive", {})
        if adaptive:
            ax.plot(
                adaptive["test_hit_rate"] * 100,
                adaptive.get("cosine_quality", adaptive.get("test_quality", 1.0)) * 100,
                "*", markersize=18, color=COLORS["quality"],
                label=f"Adaptive (θ={adaptive['best_threshold']:.2f})",
                zorder=10,
            )

    # Quality threshold line
    ax.axhline(80, color=COLORS["quality"], linestyle=":",
               linewidth=1.5, alpha=0.5, label="80% quality target")

    ax.set_xlabel("Hit Rate (%)", fontsize=13)
    ax.set_ylabel("Quality (%)", fontsize=13)
    ax.set_xlim(-2, 102)
    ax.set_ylim(50, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax.set_title(
        "Quality vs Hit Rate Tradeoff",
        fontsize=15, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "24_quality_hitrate.png", dpi=SAVE_DPI)
    plt.close(fig)
    logger.info("✓ 24_quality_hitrate.png")


# ── Plot 25: Index Comparison ────────────────────────────────────────

def plot_index_comparison(comparison: dict, output_dir: Path):
    """Compare routing performance across different index backends."""
    if not comparison:
        logger.info("⊘ No index comparison data, skipping plot 25.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    labels = []
    hit_rates = []
    qualities = []
    latency_savings = []

    for label, data in comparison.items():
        # Get the random strategy data
        strategy_data = None
        for k, v in data.items():
            if k != "meta" and isinstance(v, dict):
                strategy_data = v
                break
        if not strategy_data:
            continue

        adaptive = strategy_data.get("adaptive", {})
        if not adaptive:
            continue

        idx_type = label.split("_")[0]
        labels.append(idx_type.upper())
        hit_rates.append(adaptive.get("test_hit_rate", 0) * 100)
        qualities.append(adaptive.get("cosine_quality", adaptive.get("test_quality", 0)) * 100)
        latency_savings.append(adaptive.get("latency_saving_pct", 0))

    if not labels:
        logger.info("⊘ No comparison data with adaptive results, skipping plot 25.")
        return

    x = np.arange(len(labels))
    width = 0.5

    # Colors for each index type
    bar_colors = [
        COLORS.get(l.lower(), COLORS["hnsw"]) for l in labels
    ]

    # Hit Rate
    axes[0].bar(x, hit_rates, width, color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_title("Cache Hit Rate", fontsize=13)
    for i, v in enumerate(hit_rates):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11,
                     fontweight="bold")

    # Quality
    axes[1].bar(x, qualities, width, color=bar_colors, alpha=0.85)
    axes[1].set_ylabel("Quality (%)")
    axes[1].set_title("Routing Quality", fontsize=13)
    for i, v in enumerate(qualities):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11,
                     fontweight="bold")

    # Latency Savings
    axes[2].bar(x, latency_savings, width, color=bar_colors, alpha=0.85)
    axes[2].set_ylabel("Latency Savings (%)")
    axes[2].set_title("Latency Savings", fontsize=13)
    for i, v in enumerate(latency_savings):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11,
                     fontweight="bold")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Index Backend Comparison — Adaptive Router",
        fontsize=15, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "25_index_comparison.png", dpi=SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 25_index_comparison.png")


# ── Plot 26: Summary Dashboard ──────────────────────────────────────

def plot_summary_dashboard(routing: dict, output_dir: Path):
    """Summary table with key metrics."""
    strategies = [k for k in routing if k != "meta"]
    if not strategies:
        return

    fig, ax = plt.subplots(figsize=(16, 4 + len(strategies) * 1.2))
    ax.axis("off")

    headers = [
        "Strategy", "Cache\nSize", "Eval\nSize",
        "Best θ", "Hit Rate", "Quality",
        "Latency\nSavings", "Monetary\nSavings",
        "Adaptive θ", "Adapt\nHit Rate",
    ]

    rows = []
    for strategy in strategies:
        data = routing[strategy]
        sweep = data.get("threshold_sweep", [])
        adaptive = data.get("adaptive", {})

        if sweep:
            best = max(sweep, key=lambda x: x["latency_saving_pct"])
            best_thresh = f"{best['threshold']:.2f}"
            best_hr = f"{best['hit_rate']:.1%}"
            best_q = f"{best.get('cosine_quality', best.get('quality', 0)):.1%}"
            best_lat = f"{best['latency_saving_pct']:.1f}%"
            best_mon = f"{best['monetary_saving_pct']:.1f}%"
        else:
            best_thresh = best_hr = best_q = best_lat = best_mon = "—"

        adapt_t = f"{adaptive.get('best_threshold', 0):.2f}" if adaptive else "—"
        adapt_hr = f"{adaptive.get('test_hit_rate', 0):.1%}" if adaptive else "—"

        rows.append([
            strategy.title(),
            str(data.get("cache_size", "?")),
            str(data.get("eval_size", "?")),
            best_thresh, best_hr, best_q,
            best_lat, best_mon,
            adapt_t, adapt_hr,
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#6c63ff")
        cell.set_text_props(color="white", fontweight="bold")

    # Style rows alternating
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor("#2a2a3e" if i % 2 == 0 else "#333355")
            cell.set_edgecolor("#444466")

    n_queries = routing.get("meta", {}).get("total_queries", "?")
    ax.set_title(
        f"Phase 3 Routing Summary — {n_queries} queries",
        fontsize=15, fontweight="bold", pad=20,
        color="#ccccdd",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "26_routing_summary.png", dpi=SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 26_routing_summary.png")


# ── Plot 27: Pareto Front ────────────────────────────────────────────

def plot_pareto_front(routing: dict, output_dir: Path):
    """Quality vs Latency Savings Pareto front, colored by threshold."""
    fig, ax = plt.subplots(figsize=(10, 7))

    strategies = {"random": ("o", "#06d6a0"), "frequency": ("s", "#ffa62f")}

    for strategy, (marker, color) in strategies.items():
        data = routing.get(strategy, {})
        sweep = data.get("threshold_sweep", [])
        if not sweep:
            continue

        thresholds = [s["threshold"] for s in sweep]
        lat_savings = [s["latency_saving_pct"] for s in sweep]
        quality_key = "cosine_quality" if "cosine_quality" in sweep[0] else "quality"
        quality = [s.get(quality_key, 1.0) * 100 for s in sweep]

        # Scatter colored by threshold
        sc = ax.scatter(
            lat_savings, quality,
            c=thresholds, cmap="viridis", marker=marker,
            s=80, edgecolors="white", linewidth=0.5, alpha=0.85,
            label=f"{strategy} fill",
            zorder=5,
        )
        ax.plot(lat_savings, quality, color=color, alpha=0.3, linewidth=1, zorder=3)

        # Annotate key thresholds
        for s in sweep:
            t = s["threshold"]
            if t in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
                ax.annotate(
                    f"θ={t}",
                    (s["latency_saving_pct"], s.get(quality_key, 1.0) * 100),
                    fontsize=7, color="#aaaacc", ha="left",
                    xytext=(5, 3), textcoords="offset points",
                )

        # Mark Pareto-optimal points (non-dominated)
        pareto_idx = _pareto_front_indices(lat_savings, quality)
        pareto_lat = [lat_savings[i] for i in pareto_idx]
        pareto_q = [quality[i] for i in pareto_idx]
        ax.scatter(
            pareto_lat, pareto_q,
            facecolors="none", edgecolors="#ef476f", linewidths=2,
            s=150, zorder=6, label="Pareto optimal" if strategy == "random" else None,
        )

        # Mark adaptive operating point
        adaptive = data.get("adaptive", {})
        if adaptive:
            adapt_lat = adaptive.get("latency_saving_pct", 0)
            adapt_q = adaptive.get(quality_key, adaptive.get("quality", 0)) * 100
            ax.plot(
                adapt_lat, adapt_q, "*",
                markersize=20, color="#ef476f", markeredgecolor="white",
                markeredgewidth=1, zorder=7,
                label=f"Adaptive (θ={adaptive.get('best_threshold', '?'):.2f})"
                if strategy == "random" else None,
            )

    # Shade sweet spot region
    ax.axhspan(80, 86, alpha=0.08, color="#06d6a0", zorder=1)
    ax.annotate(
        "Sweet spot\n(θ ∈ [0.7, 0.9])", xy=(75, 83),
        fontsize=9, color="#06d6a0", style="italic", ha="center",
    )

    # 80% quality target line
    ax.axhline(80, color=COLORS["quality"], linestyle=":", alpha=0.5, linewidth=1)
    ax.annotate(
        "80% quality target", xy=(5, 80.5),
        fontsize=8, color=COLORS["quality"], alpha=0.6,
    )

    cb = fig.colorbar(sc, ax=ax, label="Threshold (L2²)", pad=0.02)
    cb.ax.yaxis.label.set_color("#ccccdd")
    cb.ax.tick_params(colors="#aaaacc")

    ax.set_xlabel("Latency Savings (%)", fontsize=13)
    ax.set_ylabel("Cosine Quality (%)", fontsize=13)
    ax.set_title("Pareto Front — Quality vs Cost Savings", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "27_pareto_front.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 27_pareto_front.png")


def _pareto_front_indices(x_vals, y_vals):
    """Find Pareto-optimal indices (maximize both x and y)."""
    n = len(x_vals)
    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if x_vals[j] >= x_vals[i] and y_vals[j] >= y_vals[i] and (
                x_vals[j] > x_vals[i] or y_vals[j] > y_vals[i]
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — Routing Evaluation Visualizations"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/routing/",
        help="Directory with routing JSON results",
    )
    parser.add_argument(
        "--output", type=str, default="results/figures/",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    routing, comparison = load_results(results_dir)

    if not routing:
        logger.error("No routing results found. Run 06_run_routing_eval.py first.")
        sys.exit(1)

    plot_threshold_sensitivity(routing, output_dir)
    plot_cost_breakdown(routing, output_dir)
    plot_distance_distribution(routing, output_dir)
    plot_quality_hitrate(routing, output_dir)
    plot_index_comparison(comparison, output_dir)
    plot_summary_dashboard(routing, output_dir)
    plot_pareto_front(routing, output_dir)

    logger.info(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
