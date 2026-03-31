#!/usr/bin/env python3
"""
Phase 4b — Workload concentration sweep visualizations.

Generates the key research figures:
  36 — Phase Diagram: hit rate vs gamma per policy
  37 — Gap to Oracle: regret vs gamma per policy
  38 — Coverage vs Hit Rate: Pareto tradeoff colored by gamma
  39 — Adaptive Alpha Trajectory: how alpha responds to workload
  40 — Diversity vs Gamma: verifying the workload generator works

Usage::

    python scripts/12_visualize_sweep.py \\
        --results results/eviction_sweep/sweep_results.json \\
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
    "adaptive": "#8B5CF6",
    "oracle": "#C44E52",
}

POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "semantic": "Semantic",
    "adaptive": "Adaptive (Ours)",
    "oracle": "Oracle (Upper Bound)",
}

POLICY_MARKERS = {
    "lru": "o",
    "lfu": "s",
    "semantic": "D",
    "adaptive": "*",
    "oracle": "^",
}


def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, output_dir, name):
    p = output_dir / f"{name}.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {p}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════
# Figure 36: Phase Diagram — hit rate vs gamma
# ═════════════════════════════════════════════════════════════════

def plot_phase_diagram(results: dict, output_dir: Path):
    """The money plot: hit rate vs workload concentration per policy."""
    agg = results["aggregated"]
    config = results["config"]
    policies = config["policies"]
    gammas = config["gamma_values"]
    cache_sizes = config["cache_sizes_pct"]

    n_sizes = len(cache_sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 5), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    for ax_idx, cache_pct in enumerate(cache_sizes):
        ax = axes[ax_idx]
        p_key = f"{cache_pct:.2f}"

        for policy in policies:
            hrs_mean = []
            hrs_std = []
            valid_gammas = []

            for gamma in gammas:
                g_key = f"{gamma:.2f}"
                stats = agg.get(g_key, {}).get(policy, {}).get(p_key, {})
                if stats:
                    hrs_mean.append(stats["hit_rate_mean"])
                    hrs_std.append(stats["hit_rate_std"])
                    valid_gammas.append(gamma)

            if not valid_gammas:
                continue

            color = POLICY_COLORS.get(policy, "#333")
            label = POLICY_LABELS.get(policy, policy)
            marker = POLICY_MARKERS.get(policy, "o")
            lw = 2.5 if policy == "adaptive" else 1.8
            ls = "--" if policy == "oracle" else "-"

            ax.plot(
                valid_gammas, hrs_mean, color=color, label=label,
                marker=marker, markersize=7, linewidth=lw,
                linestyle=ls, markevery=1,
            )
            ax.fill_between(
                valid_gammas,
                np.array(hrs_mean) - np.array(hrs_std),
                np.array(hrs_mean) + np.array(hrs_std),
                alpha=0.15, color=color,
            )

        ax.set_xlabel("γ (Workload Concentration)")
        ax.set_title(f"Cache Size = {cache_pct:.0%}")
        ax.set_xlim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

        # Mark crossover region
        ax.axvspan(0.3, 0.5, alpha=0.05, color="green")

    axes[0].set_ylabel("Hit Rate")
    axes[-1].legend(loc="best", framealpha=0.9)

    fig.suptitle(
        "Eviction Policy Performance vs. Workload Concentration",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_dir, "36_phase_diagram")


# ═════════════════════════════════════════════════════════════════
# Figure 37: Gap to Oracle (regret)
# ═════════════════════════════════════════════════════════════════

def plot_gap_to_oracle(results: dict, output_dir: Path):
    """How far each policy is from Oracle at each gamma."""
    agg = results["aggregated"]
    config = results["config"]
    policies = [p for p in config["policies"] if p != "oracle"]
    gammas = config["gamma_values"]
    cache_pct = config["cache_sizes_pct"][0]
    p_key = f"{cache_pct:.2f}"

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get oracle hit rates
    oracle_hrs = {}
    for gamma in gammas:
        g_key = f"{gamma:.2f}"
        stats = agg.get(g_key, {}).get("oracle", {}).get(p_key, {})
        if stats:
            oracle_hrs[gamma] = stats["hit_rate_mean"]

    for policy in policies:
        gaps = []
        valid_gammas = []

        for gamma in gammas:
            g_key = f"{gamma:.2f}"
            stats = agg.get(g_key, {}).get(policy, {}).get(p_key, {})
            if stats and gamma in oracle_hrs:
                gap = oracle_hrs[gamma] - stats["hit_rate_mean"]
                gaps.append(gap)
                valid_gammas.append(gamma)

        if not valid_gammas:
            continue

        color = POLICY_COLORS.get(policy, "#333")
        label = POLICY_LABELS.get(policy, policy)
        marker = POLICY_MARKERS.get(policy, "o")
        lw = 2.5 if policy == "adaptive" else 1.8

        ax.plot(
            valid_gammas, gaps, color=color, label=label,
            marker=marker, markersize=7, linewidth=lw,
        )

    ax.set_xlabel("γ (Workload Concentration)")
    ax.set_ylabel("Hit Rate Gap to Oracle")
    ax.set_title(f"Regret vs. Oracle (Cache = {cache_pct:.0%})")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="best", framealpha=0.9)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, output_dir, "37_gap_to_oracle")


# ═════════════════════════════════════════════════════════════════
# Figure 38: Coverage vs Hit Rate (Pareto)
# ═════════════════════════════════════════════════════════════════

def plot_coverage_vs_hitrate(results: dict, output_dir: Path):
    """Scatter: coverage (x) vs hit rate (y), colored by policy, sized by gamma."""
    agg = results["aggregated"]
    config = results["config"]
    policies = config["policies"]
    gammas = config["gamma_values"]
    cache_pct = config["cache_sizes_pct"][0]
    p_key = f"{cache_pct:.2f}"

    fig, ax = plt.subplots(figsize=(8, 6))

    for policy in policies:
        coverages = []
        hit_rates = []
        gamma_vals = []

        for gamma in gammas:
            g_key = f"{gamma:.2f}"
            stats = agg.get(g_key, {}).get(policy, {}).get(p_key, {})
            if stats:
                coverages.append(stats["coverage_mean"])
                hit_rates.append(stats["hit_rate_mean"])
                gamma_vals.append(gamma)

        if not coverages:
            continue

        color = POLICY_COLORS.get(policy, "#333")
        label = POLICY_LABELS.get(policy, policy)
        marker = POLICY_MARKERS.get(policy, "o")

        # Size proportional to gamma (larger = more concentrated)
        sizes = [40 + 120 * g for g in gamma_vals]

        ax.scatter(
            coverages, hit_rates, c=color, s=sizes,
            marker=marker, label=label, alpha=0.7,
            edgecolors="white", linewidths=0.5,
        )
        # Connect points with a line
        ax.plot(coverages, hit_rates, color=color, alpha=0.3, linewidth=1)

    ax.set_xlabel("Semantic Coverage (Avg L2² to Nearest Cached — Lower = Better)")
    ax.set_ylabel("Hit Rate")
    ax.set_title(f"Coverage vs Hit Rate Tradeoff (Cache = {cache_pct:.0%})")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_dir, "38_coverage_vs_hitrate")


# ═════════════════════════════════════════════════════════════════
# Figure 39: Adaptive Alpha Trajectory
# ═════════════════════════════════════════════════════════════════

def plot_alpha_trajectory(results: dict, output_dir: Path):
    """Show how the adaptive policy's alpha responds to different gammas."""
    sweep = results["sweep_results"]

    # Find adaptive runs at first cache size, first seed
    config = results["config"]
    cache_pct = config["cache_sizes_pct"][0]
    seed = config["seeds"][0] if isinstance(config.get("seeds"), list) else 42
    gammas = config["gamma_values"]

    adaptive_runs = [
        r for r in sweep
        if r.get("policy") == "adaptive"
        and r.get("cache_size_pct") == cache_pct
        and r.get("seed") == seed
        and "error" not in r
    ]

    if not adaptive_runs:
        logger.warning("No adaptive runs found, skipping alpha trajectory plot")
        return

    # Select a subset of gammas to show (avoid clutter)
    show_gammas = [0.0, 0.3, 0.5, 0.7, 1.0]
    show_gammas = [g for g in show_gammas if g in gammas]
    if not show_gammas:
        show_gammas = gammas[:5]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis

    for run in adaptive_runs:
        gamma = run["gamma"]
        if gamma not in show_gammas:
            continue

        history = run.get("alpha_history", [])
        if not history:
            continue

        steps = [h["adaptation_step"] for h in history]
        alphas = [h["new_alpha"] for h in history]
        color = cmap(gamma)

        ax.plot(
            steps, alphas, color=color,
            label=f"γ={gamma:.1f}", linewidth=2, alpha=0.8,
        )

    ax.set_xlabel("Adaptation Step")
    ax.set_ylabel("α (Recency Weight)")
    ax.set_title("Adaptive Policy: α Trajectory by Workload Concentration")
    ax.legend(loc="best", framealpha=0.9)

    # Annotate regions
    ax.axhspan(0.3, 0.8, alpha=0.05, color="green")
    ax.text(
        0.02, 0.5, "Redundancy-\nfocused",
        transform=ax.transAxes, fontsize=9, color="green", alpha=0.7,
        va="center",
    )
    ax.axhspan(1.5, 2.5, alpha=0.05, color="blue")
    ax.text(
        0.02, 0.9, "Recency-\nfocused\n(LRU-like)",
        transform=ax.transAxes, fontsize=9, color="blue", alpha=0.7,
        va="center",
    )

    fig.tight_layout()
    _save(fig, output_dir, "39_alpha_trajectory")


# ═════════════════════════════════════════════════════════════════
# Figure 40: Diversity vs Gamma (sanity check)
# ═════════════════════════════════════════════════════════════════

def plot_diversity_vs_gamma(results: dict, output_dir: Path):
    """Verify that gamma actually controls diversity as expected."""
    agg = results["aggregated"]
    config = results["config"]
    gammas = config["gamma_values"]
    # Use any policy — diversity is a workload property, not policy-dependent
    policy = config["policies"][0]
    cache_pct = config["cache_sizes_pct"][0]
    p_key = f"{cache_pct:.2f}"

    divs = []
    valid_gammas = []

    for gamma in gammas:
        g_key = f"{gamma:.2f}"
        stats = agg.get(g_key, {}).get(policy, {}).get(p_key, {})
        if stats and "diversity_mean" in stats:
            divs.append(stats["diversity_mean"])
            valid_gammas.append(gamma)

    if not valid_gammas:
        logger.warning("No diversity data, skipping plot 40")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(valid_gammas, divs, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax.set_xlabel("γ (Workload Concentration Parameter)")
    ax.set_ylabel("Normalized Cluster Entropy (Diversity)")
    ax.set_title("Workload Diversity vs. Concentration Parameter")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Expected: monotonically decreasing
    ax.annotate(
        "Uniform\n(diverse)", xy=(0.0, divs[0] if divs else 0.8),
        fontsize=10, ha="center",
        xytext=(0.15, divs[0] + 0.08 if divs else 0.9),
        arrowprops=dict(arrowstyle="->", color="#666"),
    )
    if len(divs) > 1:
        ax.annotate(
            "Concentrated\n(few topics)", xy=(1.0, divs[-1]),
            fontsize=10, ha="center",
            xytext=(0.85, divs[-1] + 0.12),
            arrowprops=dict(arrowstyle="->", color="#666"),
        )

    fig.tight_layout()
    _save(fig, output_dir, "40_diversity_vs_gamma")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4b: Sweep visualization"
    )
    parser.add_argument(
        "--results", default="results/eviction_sweep/sweep_results.json",
    )
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    setup_style()
    results = load_results(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_diagram(results, output_dir)
    plot_gap_to_oracle(results, output_dir)
    plot_coverage_vs_hitrate(results, output_dir)
    plot_alpha_trajectory(results, output_dir)
    plot_diversity_vs_gamma(results, output_dir)

    logger.info(f"\nAll sweep figures saved to {output_dir}")


if __name__ == "__main__":
    main()