#!/usr/bin/env python3
"""
05_visualize_benchmarks.py — Phase 2: Visualize index benchmark results.

Generates publication-quality plots from Phase 2 JSON benchmark results.

Usage:
    python scripts/05_visualize_benchmarks.py \
        --results-dir results/benchmarks/ \
        --output results/figures/

Figures generated:
    11_recall_vs_latency.png     — Recall@10 vs P50 latency tradeoff
    12_build_time.png            — Build time comparison
    13_memory_usage.png          — Memory footprint comparison
    14_throughput_qps.png        — Query throughput comparison
    15_latency_distribution.png  — P50/P95/P99 latency breakdown
    16_parameter_sensitivity.png — HNSW efSearch & IVF nprobe sweeps
    17_scalability.png           — Metrics across dataset scales
    18_benchmark_summary.png     — Summary dashboard
"""

import json
import logging
import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Style ---
COLORS = {
    "flat": "#6366f1",   # indigo
    "hnsw": "#10b981",   # emerald
    "ivf": "#f59e0b",    # amber
    "lsh": "#ef4444",    # red
}
INDEX_LABELS = {"flat": "Flat (Exact)", "hnsw": "HNSW", "ivf": "IVF", "lsh": "LSH"}

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#475569",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all benchmark JSON files, keyed by scale label."""
    results = {}
    for f in sorted(results_dir.glob("index_benchmark_*.json")):
        label = f.stem.replace("index_benchmark_", "")
        with open(f) as fh:
            data = json.load(fh)
        # Filter out entries with errors
        data = [r for r in data if "error" not in r]
        if data:
            results[label] = data
            logger.info(f"Loaded {label}: {len(data)} configurations")
    return results


def _param_label(r: dict) -> str:
    """Short parameter label for legends."""
    p = r.get("params", {})
    t = r["index_type"]
    if t == "flat":
        return "Flat"
    elif t == "hnsw":
        return f"M={p.get('M')},ef={p.get('efSearch')}"
    elif t == "ivf":
        return f"nl={p.get('nlist')},np={p.get('nprobe')}"
    elif t == "lsh":
        return f"nb={p.get('nbits')}"
    return str(p)


# ────────────────────────────────────────────────────────
# Figure 11: Recall vs Latency Tradeoff
# ────────────────────────────────────────────────────────
def plot_recall_vs_latency(results: dict, output_dir: Path):
    """Scatter plot: recall@10 vs P50 latency for each index config."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use the largest available scale
    scale = list(results.keys())[-1]
    data = results[scale]

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = [r for r in data if r["index_type"] == idx_type]
        if not entries:
            continue

        recalls = [r.get("recall_at_10", 0) for r in entries]
        latencies = [r["search_latency_ms"]["p50"] for r in entries]

        ax.scatter(
            latencies, recalls,
            c=COLORS[idx_type], s=100, alpha=0.85,
            label=INDEX_LABELS[idx_type], edgecolors="white", linewidth=0.5,
            zorder=5
        )

        # Annotate best recall point
        best_idx = int(np.argmax(recalls))
        ax.annotate(
            _param_label(entries[best_idx]),
            (latencies[best_idx], recalls[best_idx]),
            fontsize=7, color=COLORS[idx_type],
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("P50 Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Recall vs Latency Tradeoff — {scale} scale")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "11_recall_vs_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 11_recall_vs_latency.png")


# ────────────────────────────────────────────────────────
# Figure 12: Build Time Comparison
# ────────────────────────────────────────────────────────
def plot_build_time(results: dict, output_dir: Path):
    """Bar chart: build time for best config of each index type."""
    fig, ax = plt.subplots(figsize=(9, 6))

    scale = list(results.keys())[-1]
    data = results[scale]

    idx_types = []
    build_times = []
    colors = []

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = [r for r in data if r["index_type"] == idx_type]
        if not entries:
            continue
        # Pick config with best recall
        best = max(entries, key=lambda r: r.get("recall_at_10", 0))
        idx_types.append(f"{INDEX_LABELS[idx_type]}\n{_param_label(best)}")
        build_times.append(best["build_time_s"])
        colors.append(COLORS[idx_type])

    bars = ax.bar(idx_types, build_times, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, build_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.3f}s", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#e2e8f0")

    ax.set_ylabel("Build Time (seconds)")
    ax.set_title(f"Index Build Time — {scale} scale (best recall config)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "12_build_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 12_build_time.png")


# ────────────────────────────────────────────────────────
# Figure 13: Memory Usage Comparison
# ────────────────────────────────────────────────────────
def plot_memory_usage(results: dict, output_dir: Path):
    """Bar chart: memory usage for best config of each index type."""
    fig, ax = plt.subplots(figsize=(9, 6))

    scale = list(results.keys())[-1]
    data = results[scale]

    idx_types = []
    memory_mbs = []
    colors = []

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = [r for r in data if r["index_type"] == idx_type]
        if not entries:
            continue
        best = max(entries, key=lambda r: r.get("recall_at_10", 0))
        idx_types.append(f"{INDEX_LABELS[idx_type]}\n{_param_label(best)}")
        memory_mbs.append(best["memory_mb"])
        colors.append(COLORS[idx_type])

    bars = ax.bar(idx_types, memory_mbs, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    for bar, m in zip(bars, memory_mbs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.1f} MB", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#e2e8f0")

    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"Index Memory Usage — {scale} scale (best recall config)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "13_memory_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 13_memory_usage.png")


# ────────────────────────────────────────────────────────
# Figure 14: Throughput Comparison
# ────────────────────────────────────────────────────────
def plot_throughput(results: dict, output_dir: Path):
    """Bar chart: throughput (QPS) for all configs grouped by index type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    scale = list(results.keys())[-1]
    data = results[scale]

    labels = []
    qps_values = []
    colors_list = []

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = sorted(
            [r for r in data if r["index_type"] == idx_type],
            key=lambda r: r.get("recall_at_10", 0),
        )
        for r in entries:
            labels.append(f"{idx_type}\n{_param_label(r)}")
            qps_values.append(r["throughput_qps"])
            colors_list.append(COLORS[idx_type])

    x = np.arange(len(labels))
    bars = ax.bar(x, qps_values, color=colors_list, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Throughput (queries/sec)")
    ax.set_title(f"Search Throughput — {scale} scale")
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[t], label=INDEX_LABELS[t])
                       for t in ["flat", "hnsw", "ivf", "lsh"]]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / "14_throughput_qps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 14_throughput_qps.png")


# ────────────────────────────────────────────────────────
# Figure 15: Latency Distribution
# ────────────────────────────────────────────────────────
def plot_latency_distribution(results: dict, output_dir: Path):
    """Grouped bar chart: P50, P95, P99 latency for best config per index."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scale = list(results.keys())[-1]
    data = results[scale]

    idx_types = []
    p50s, p95s, p99s = [], [], []
    colors_list = []

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = [r for r in data if r["index_type"] == idx_type]
        if not entries:
            continue
        best = max(entries, key=lambda r: r.get("recall_at_10", 0))
        idx_types.append(f"{INDEX_LABELS[idx_type]}\n{_param_label(best)}")
        lat = best["search_latency_ms"]
        p50s.append(lat["p50"])
        p95s.append(lat["p95"])
        p99s.append(lat["p99"])
        colors_list.append(COLORS[idx_type])

    x = np.arange(len(idx_types))
    width = 0.25

    b1 = ax.bar(x - width, p50s, width, label="P50", color="#6366f1", alpha=0.85)
    b2 = ax.bar(x, p95s, width, label="P95", color="#f59e0b", alpha=0.85)
    b3 = ax.bar(x + width, p99s, width, label="P99", color="#ef4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(idx_types, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Search Latency Distribution — {scale} scale (best recall config)")
    ax.legend(framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "15_latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 15_latency_distribution.png")


# ────────────────────────────────────────────────────────
# Figure 16: Parameter Sensitivity
# ────────────────────────────────────────────────────────
def plot_parameter_sensitivity(results: dict, output_dir: Path):
    """Line plots: HNSW efSearch sweep and IVF nprobe sweep."""
    scale = list(results.keys())[-1]
    data = results[scale]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- HNSW: efSearch vs recall (fixed M) ---
    ax = axes[0]
    hnsw_entries = [r for r in data if r["index_type"] == "hnsw"]

    # Group by M
    m_groups = {}
    for r in hnsw_entries:
        m = r["params"]["M"]
        if m not in m_groups:
            m_groups[m] = []
        m_groups[m].append(r)

    m_colors = {16: "#6366f1", 32: "#10b981", 64: "#f59e0b"}
    for m, entries in sorted(m_groups.items()):
        entries_sorted = sorted(entries, key=lambda r: r["params"]["efSearch"])
        ef_vals = [r["params"]["efSearch"] for r in entries_sorted]
        recalls = [r.get("recall_at_10", 0) for r in entries_sorted]
        latencies = [r["search_latency_ms"]["p50"] for r in entries_sorted]

        color = m_colors.get(m, "#94a3b8")
        ax.plot(ef_vals, recalls, "o-", color=color, label=f"M={m}",
                markersize=8, linewidth=2, alpha=0.85)

    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@10")
    ax.set_title("HNSW: efSearch vs Recall@10")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.005)

    # --- IVF: nprobe vs recall (fixed nlist) ---
    ax = axes[1]
    ivf_entries = [r for r in data if r["index_type"] == "ivf"]

    nlist_groups = {}
    for r in ivf_entries:
        nlist = r["params"]["nlist"]
        if nlist not in nlist_groups:
            nlist_groups[nlist] = []
        nlist_groups[nlist].append(r)

    nlist_colors = {64: "#6366f1", 256: "#10b981", 1024: "#f59e0b", 4096: "#ef4444"}
    for nlist, entries in sorted(nlist_groups.items()):
        entries_sorted = sorted(entries, key=lambda r: r["params"]["nprobe"])
        nprobe_vals = [r["params"]["nprobe"] for r in entries_sorted]
        recalls = [r.get("recall_at_10", 0) for r in entries_sorted]

        color = nlist_colors.get(nlist, "#94a3b8")
        ax.plot(nprobe_vals, recalls, "s-", color=color, label=f"nlist={nlist}",
                markersize=8, linewidth=2, alpha=0.85)

    ax.set_xlabel("nprobe")
    ax.set_ylabel("Recall@10")
    ax.set_title("IVF: nprobe vs Recall@10")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Parameter Sensitivity — {scale} scale", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "16_parameter_sensitivity.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 16_parameter_sensitivity.png")


# ────────────────────────────────────────────────────────
# Figure 17: Scalability Across Dataset Sizes
# ────────────────────────────────────────────────────────
def plot_scalability(results: dict, output_dir: Path):
    """Line plots: latency, recall, build time, memory across scales."""
    if len(results) < 2:
        logger.warning("Need ≥2 scales for scalability plot, skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # For each index type, pick best-recall config and track across scales
    scale_order = []
    scale_sizes = {}
    for label, data in results.items():
        if data:
            size = data[0].get("dataset_size", 0)
            scale_order.append((size, label))
            scale_sizes[label] = size
    scale_order.sort()
    ordered_labels = [s[1] for s in scale_order]
    ordered_sizes = [s[0] for s in scale_order]

    metrics = [
        (axes[0, 0], "search_latency_ms", "P50 Latency (ms)", lambda r: r["search_latency_ms"]["p50"]),
        (axes[0, 1], "recall_at_10", "Recall@10", lambda r: r.get("recall_at_10", 0)),
        (axes[1, 0], "build_time_s", "Build Time (s)", lambda r: r["build_time_s"]),
        (axes[1, 1], "memory_mb", "Memory (MB)", lambda r: r["memory_mb"]),
    ]

    for ax, metric_key, ylabel, extractor in metrics:
        for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
            values = []
            valid_sizes = []

            for label in ordered_labels:
                entries = [r for r in results[label] if r["index_type"] == idx_type]
                if not entries:
                    continue
                # Pick best recall config
                best = max(entries, key=lambda r: r.get("recall_at_10", 0))
                values.append(extractor(best))
                valid_sizes.append(scale_sizes[label])

            if values:
                ax.plot(valid_sizes, values, "o-", color=COLORS[idx_type],
                        label=INDEX_LABELS[idx_type], markersize=7,
                        linewidth=2, alpha=0.85)

        ax.set_xlabel("Dataset Size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("Scalability Analysis — Best Config per Index", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "17_scalability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 17_scalability.png")


# ────────────────────────────────────────────────────────
# Figure 18: Summary Dashboard
# ────────────────────────────────────────────────────────
def plot_summary_dashboard(results: dict, output_dir: Path):
    """Summary dashboard with key metrics in a table-like layout."""
    scale = list(results.keys())[-1]
    data = results[scale]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    # Build summary table
    headers = ["Index", "Config", "Build (s)", "Mem (MB)", "P50 (ms)",
               "P99 (ms)", "QPS", "R@1", "R@5", "R@10"]
    rows = []

    for idx_type in ["flat", "hnsw", "ivf", "lsh"]:
        entries = [r for r in data if r["index_type"] == idx_type]
        if not entries:
            continue
        best = max(entries, key=lambda r: r.get("recall_at_10", 0))
        rows.append([
            INDEX_LABELS[idx_type],
            _param_label(best),
            f"{best['build_time_s']:.3f}",
            f"{best['memory_mb']:.1f}",
            f"{best['search_latency_ms']['p50']:.3f}",
            f"{best['search_latency_ms']['p99']:.3f}",
            f"{best['throughput_qps']:.0f}",
            f"{best.get('recall_at_1', 0):.4f}",
            f"{best.get('recall_at_5', 0):.4f}",
            f"{best.get('recall_at_10', 0):.4f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#6366f1")
        cell.set_text_props(color="white", fontweight="bold")

    # Style rows with index-specific colors
    for i, row in enumerate(rows):
        idx_name = row[0].lower().replace(" (exact)", "")
        color = COLORS.get(idx_name, "#94a3b8")
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#e2e8f0")
            cell.set_edgecolor("#475569")

    ax.set_title(
        f"  Phase 2 Benchmark Summary — {scale} scale  ",
        fontsize=14, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "18_benchmark_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ 18_benchmark_summary.png")


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
@click.command()
@click.option("--results-dir", type=click.Path(exists=True),
              default="results/benchmarks/",
              help="Directory containing benchmark JSON files.")
@click.option("--output", type=click.Path(), default="results/figures/",
              help="Output directory for PNG figures.")
def main(results_dir, output):
    """Generate Phase 2 benchmark visualizations."""
    results_dir = Path(results_dir)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    results = load_results(results_dir)
    if not results:
        logger.error("No benchmark results found!")
        sys.exit(1)

    logger.info(f"Scales available: {list(results.keys())}")

    # Generate all plots
    plot_recall_vs_latency(results, output_dir)
    plot_build_time(results, output_dir)
    plot_memory_usage(results, output_dir)
    plot_throughput(results, output_dir)
    plot_latency_distribution(results, output_dir)
    plot_parameter_sensitivity(results, output_dir)
    plot_scalability(results, output_dir)
    plot_summary_dashboard(results, output_dir)

    logger.info(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
