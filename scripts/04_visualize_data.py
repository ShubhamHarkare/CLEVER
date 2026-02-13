#!/usr/bin/env python3
"""
04_visualize_data.py — Phase 1 Data & Embedding Visualizations.

Generates a comprehensive set of plots for understanding the LMSYS-Chat-1M
dataset, preprocessing effects, and embedding space structure.

Outputs are saved to results/figures/ as PNG files (Great Lakes compatible).

Usage:
    python scripts/04_visualize_data.py --data-dir data/ --embed-dir results/embeddings/ --output results/figures/
"""

import logging
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (Great Lakes compatible)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Style Configuration ──────────────────────────────────────────────
COLORS = {
    "primary": "#4361EE",
    "secondary": "#7209B7",
    "accent": "#F72585",
    "success": "#06D6A0",
    "warning": "#FFD166",
    "dark": "#2B2D42",
    "light": "#EDF2F4",
    "gradient": ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#FF6B6B"],
}

PALETTE = [
    "#4361EE", "#F72585", "#7209B7", "#06D6A0", "#FFD166",
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D5DBDB",
    "#C39BD3", "#73C6B6", "#F9E79F", "#D7BDE2", "#A9CCE3",
]


def set_style():
    """Set a publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "axes.edgecolor": "#D0D5DD",
        "axes.labelcolor": "#344054",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#D0D5DD",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


# ── Plot Functions ───────────────────────────────────────────────────


def plot_query_length_distribution(df_raw, df_processed, output_dir):
    """Plot token count distribution before and after filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before filtering
    raw_tokens = df_raw["query_text"].str.split().str.len()
    axes[0].hist(raw_tokens, bins=80, color=COLORS["primary"], alpha=0.8,
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(3, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Min (3 tokens)")
    axes[0].axvline(512, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Max (512 tokens)")
    axes[0].set_title("Query Length — Before Filtering", fontweight="bold")
    axes[0].set_xlabel("Token Count")
    axes[0].set_ylabel("Number of Queries")
    axes[0].legend()
    axes[0].set_xlim(0, min(600, raw_tokens.max() + 10))

    # After filtering
    axes[1].hist(df_processed["token_count"], bins=80, color=COLORS["success"], alpha=0.8,
                 edgecolor="white", linewidth=0.5)
    axes[1].set_title("Query Length — After Filtering", fontweight="bold")
    axes[1].set_xlabel("Token Count")
    axes[1].set_ylabel("Number of Queries")

    # Add statistics box
    stats_text = (
        f"Mean: {df_processed['token_count'].mean():.1f}\n"
        f"Median: {df_processed['token_count'].median():.0f}\n"
        f"Std: {df_processed['token_count'].std():.1f}"
    )
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    fig.suptitle("LMSYS-Chat-1M: Query Length Distribution", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "01_query_length_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_duplicate_frequency(df_processed, output_dir):
    """Plot distribution of query frequencies (duplicates)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    freq = df_processed["frequency"]

    # Histogram of frequencies
    max_freq = min(freq.max(), 50)
    axes[0].hist(freq[freq <= max_freq], bins=range(1, int(max_freq) + 2),
                 color=COLORS["secondary"], alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Query Frequency Distribution", fontweight="bold")
    axes[0].set_xlabel("Number of Occurrences")
    axes[0].set_ylabel("Number of Unique Queries")
    axes[0].set_yscale("log")

    # Stats
    unique_count = (freq == 1).sum()
    dup_count = (freq > 1).sum()
    stats_text = (
        f"Unique (freq=1): {unique_count} ({unique_count/len(freq)*100:.1f}%)\n"
        f"Duplicated: {dup_count} ({dup_count/len(freq)*100:.1f}%)\n"
        f"Max frequency: {freq.max()}"
    )
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    # Top-N most frequent queries
    top_n = 15
    top_queries = df_processed.nlargest(top_n, "frequency")
    labels = [q[:50] + "..." if len(q) > 50 else q for q in top_queries["query_text"]]
    bars = axes[1].barh(range(top_n), top_queries["frequency"].values,
                        color=COLORS["gradient"][0], alpha=0.8)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top {top_n} Most Frequent Queries", fontweight="bold")
    axes[1].set_xlabel("Frequency")

    fig.suptitle("LMSYS-Chat-1M: Query Duplication Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "02_duplicate_frequency.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_lmsys_model_distribution(df, output_dir):
    """Plot which LLMs users were chatting with."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_counts = df["model"].value_counts()

    # Bar chart — Top 15 models
    top_n = min(15, len(model_counts))
    top_models = model_counts.head(top_n)
    colors = PALETTE[:top_n]
    bars = axes[0].barh(range(top_n), top_models.values, color=colors, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_models.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Top {top_n} LLM Models", fontweight="bold")
    axes[0].set_xlabel("Number of Queries")

    # Add percentage labels
    total = model_counts.sum()
    for i, (v, name) in enumerate(zip(top_models.values, top_models.index)):
        axes[0].text(v + total * 0.005, i, f"{v/total*100:.1f}%", va="center", fontsize=8)

    # Pie chart — model family grouping
    def get_model_family(name):
        name = name.lower()
        if "vicuna" in name:
            return "Vicuna"
        elif "llama" in name:
            return "LLaMA"
        elif "alpaca" in name:
            return "Alpaca"
        elif "chatglm" in name:
            return "ChatGLM"
        elif "koala" in name:
            return "Koala"
        elif "oasst" in name or "pythia" in name:
            return "OpenAssistant"
        elif "dolly" in name:
            return "Dolly"
        elif "fastchat" in name or "t5" in name:
            return "FastChat-T5"
        elif "gpt" in name:
            return "GPT"
        elif "palm" in name or "bard" in name:
            return "PaLM/Bard"
        elif "claude" in name:
            return "Claude"
        elif "mpt" in name:
            return "MPT"
        elif "stablelm" in name:
            return "StableLM"
        elif "guanaco" in name:
            return "Guanaco"
        else:
            return "Other"

    df_copy = df.copy()
    df_copy["model_family"] = df_copy["model"].apply(get_model_family)
    family_counts = df_copy["model_family"].value_counts()

    wedges, texts, autotexts = axes[1].pie(
        family_counts.values, labels=family_counts.index,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=PALETTE[:len(family_counts)],
        startangle=90, pctdistance=0.8
    )
    for text in texts:
        text.set_fontsize(8)
    for text in autotexts:
        text.set_fontsize(7)
    axes[1].set_title("Model Family Distribution", fontweight="bold")

    fig.suptitle("LMSYS-Chat-1M: LLM Model Distribution", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "03_lmsys_model_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_language_distribution(df, output_dir):
    """Plot language distribution of queries."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lang_counts = df["language"].value_counts()

    # Bar chart — Top 15 languages
    top_n = min(15, len(lang_counts))
    top_langs = lang_counts.head(top_n)
    bars = axes[0].barh(range(top_n), top_langs.values,
                        color=PALETTE[:top_n], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_langs.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Top {top_n} Languages", fontweight="bold")
    axes[0].set_xlabel("Number of Queries")

    total = lang_counts.sum()
    for i, v in enumerate(top_langs.values):
        axes[0].text(v + total * 0.005, i, f"{v/total*100:.1f}%", va="center", fontsize=8)

    # English vs non-English pie
    english_count = lang_counts.get("English", 0)
    non_english = total - english_count
    axes[1].pie([english_count, non_english],
                labels=["English", "Non-English"],
                autopct="%1.1f%%",
                colors=[COLORS["primary"], COLORS["accent"]],
                startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("English vs Non-English", fontweight="bold")

    fig.suptitle("LMSYS-Chat-1M: Language Distribution", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "04_language_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_conversation_turns(df, output_dir):
    """Plot distribution of conversation turns."""
    fig, ax = plt.subplots(figsize=(10, 5))

    turns = df["num_turns"]
    max_turns = min(turns.max(), 30)
    ax.hist(turns[turns <= max_turns], bins=range(1, int(max_turns) + 2),
            color=COLORS["primary"], alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_title("Conversation Turn Count Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Number of Conversations")

    stats_text = (
        f"Mean: {turns.mean():.1f}\n"
        f"Median: {turns.median():.0f}\n"
        f"Max: {turns.max()}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    plt.tight_layout()
    path = output_dir / "05_conversation_turns.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_preprocessing_funnel(df_raw, df_processed, output_dir):
    """Show how data shrinks through each preprocessing step."""
    fig, ax = plt.subplots(figsize=(10, 5))

    raw_count = len(df_raw)
    raw_tokens = df_raw["query_text"].str.split().str.len()
    after_filter = ((raw_tokens >= 3) & (raw_tokens <= 512)).sum()
    final_count = len(df_processed)

    stages = ["Raw Queries", "After Length Filter\n(3–512 tokens)", "After Deduplication"]
    counts = [raw_count, after_filter, final_count]
    colors_bar = [COLORS["primary"], COLORS["secondary"], COLORS["success"]]

    bars = ax.bar(stages, counts, color=colors_bar, alpha=0.85,
                  edgecolor="white", linewidth=1)

    # Add count and percentage labels
    for bar, count in zip(bars, counts):
        pct = count / raw_count * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + raw_count * 0.01,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add drop arrows between bars
    for i in range(len(counts) - 1):
        drop = counts[i] - counts[i + 1]
        mid_x = (i + i + 1) / 2 + 0.5
        mid_y = (counts[i] + counts[i + 1]) / 2
        ax.annotate(f"−{drop:,}", xy=(mid_x, mid_y),
                    fontsize=9, ha="center", color=COLORS["accent"], fontweight="bold")

    ax.set_title("Data Preprocessing Funnel", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Queries")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = output_dir / "06_preprocessing_funnel.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_embedding_similarity_distribution(embeddings, output_dir, sample_size=2000):
    """Plot pairwise cosine similarity distribution among embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sample to keep computation tractable
    n = min(sample_size, len(embeddings))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sample = embeddings[idx]

    # Pairwise cosine similarities (embeddings are L2-normalized, so dot product)
    sims = sample @ sample.T
    # Get upper triangle (excluding diagonal)
    upper = sims[np.triu_indices(n, k=1)]

    # Histogram
    axes[0].hist(upper, bins=100, color=COLORS["primary"], alpha=0.8,
                 edgecolor="white", linewidth=0.3, density=True)
    axes[0].set_title("Pairwise Cosine Similarity Distribution", fontweight="bold")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Density")

    stats_text = (
        f"Mean: {upper.mean():.3f}\n"
        f"Std: {upper.std():.3f}\n"
        f"Min: {upper.min():.3f}\n"
        f"Max: {upper.max():.3f}\n"
        f"Pairs: {len(upper):,}"
    )
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    # CDF
    sorted_sims = np.sort(upper)
    cdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    axes[1].plot(sorted_sims, cdf, color=COLORS["secondary"], linewidth=2)
    axes[1].set_title("Cumulative Similarity Distribution (CDF)", fontweight="bold")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Cumulative Probability")

    # Add threshold lines for cache hit analysis
    for thresh, label in [(0.85, "τ=0.85"), (0.90, "τ=0.90"), (0.95, "τ=0.95")]:
        frac_above = (upper >= thresh).mean()
        axes[1].axvline(thresh, color=COLORS["accent"], linestyle="--", alpha=0.7, linewidth=1)
        axes[1].text(thresh + 0.005, 0.5, f"{label}\n({frac_above*100:.2f}%)",
                     fontsize=8, color=COLORS["accent"])

    fig.suptitle("LMSYS-Chat-1M: Embedding Space Similarity", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "07_embedding_similarity.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_embedding_tsne(embeddings, df, output_dir, sample_size=3000):
    """2D t-SNE visualization of embedding space, colored by model."""
    from sklearn.manifold import TSNE

    # Sample
    n = min(sample_size, len(embeddings))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sample_emb = embeddings[idx]
    sample_df = df.iloc[idx].reset_index(drop=True)

    logger.info(f"Running t-SNE on {n} embeddings (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(sample_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by model
    if "model" in sample_df.columns:
        top_models = sample_df["model"].value_counts().head(8).index.tolist()
        sample_df["model_group"] = sample_df["model"].apply(
            lambda x: x if x in top_models else "Other"
        )
        groups = sample_df["model_group"].unique()

        for i, group in enumerate(groups):
            mask = sample_df["model_group"] == group
            color = PALETTE[i % len(PALETTE)]
            alpha = 0.6 if group != "Other" else 0.15
            size = 10 if group != "Other" else 4
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                          c=color, label=group, alpha=alpha, s=size, edgecolors="none")

        axes[0].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
    else:
        axes[0].scatter(coords[:, 0], coords[:, 1], c=COLORS["primary"],
                       alpha=0.3, s=8, edgecolors="none")

    axes[0].set_title("t-SNE — Colored by LLM Model", fontweight="bold")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    # Color by language
    if "language" in sample_df.columns:
        sample_df["lang_group"] = sample_df["language"].apply(
            lambda x: x if x in ["English", "Portuguese", "Russian", "German",
                                   "Spanish", "French", "Italian", "Chinese"] else "Other"
        )
        groups = sample_df["lang_group"].unique()

        for i, group in enumerate(groups):
            mask = sample_df["lang_group"] == group
            color = PALETTE[i % len(PALETTE)]
            alpha = 0.6 if group != "Other" else 0.15
            size = 10 if group != "Other" else 4
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                          c=color, label=group, alpha=alpha, s=size, edgecolors="none")

        axes[1].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
    else:
        axes[1].scatter(coords[:, 0], coords[:, 1], c=COLORS["primary"],
                       alpha=0.3, s=8, edgecolors="none")

    axes[1].set_title("t-SNE — Colored by Language", fontweight="bold")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    fig.suptitle("LMSYS-Chat-1M: Embedding Space (t-SNE)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "08_embedding_tsne.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_embedding_norms(embeddings, output_dir):
    """Verify L2 normalization of embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    norms = np.linalg.norm(embeddings, axis=1)

    # Handle case where norms are very tightly clustered (e.g., all ≈ 1.0)
    norm_range = norms.max() - norms.min()
    if norm_range < 1e-6:
        # All norms are essentially identical — show a bar chart instead
        axes[0].bar(["L2 Norm"], [norms.mean()], color=COLORS["success"],
                    alpha=0.8, width=0.4)
        axes[0].set_ylim(0.998, 1.002)
        axes[0].set_title("L2 Norms — Perfectly Normalized ✓", fontweight="bold")
    else:
        axes[0].hist(norms, bins=50, color=COLORS["success"], alpha=0.8,
                     edgecolor="white", linewidth=0.5)
        axes[0].set_title("L2 Norm Distribution", fontweight="bold")
        axes[0].set_xlabel("L2 Norm")
    axes[0].set_ylabel("Count")
    axes[0].text(0.95, 0.95,
                 f"Mean: {norms.mean():.6f}\nStd: {norms.std():.2e}\nMin: {norms.min():.6f}\nMax: {norms.max():.6f}",
                 transform=axes[0].transAxes, va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    # Per-dimension mean and std
    dim_means = embeddings.mean(axis=0)
    dim_stds = embeddings.std(axis=0)
    axes[1].fill_between(range(embeddings.shape[1]),
                         dim_means - dim_stds, dim_means + dim_stds,
                         alpha=0.3, color=COLORS["primary"])
    axes[1].plot(range(embeddings.shape[1]), dim_means,
                color=COLORS["primary"], linewidth=0.8)
    axes[1].set_title("Per-Dimension Statistics", fontweight="bold")
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Value (mean ± std)")

    fig.suptitle("LMSYS-Chat-1M: Embedding Quality Check", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "09_embedding_norms.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_model_query_length_heatmap(df, output_dir):
    """Heatmap: average query length by model and language."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get top models and languages
    top_models = df["model"].value_counts().head(10).index.tolist()
    top_langs = df["language"].value_counts().head(8).index.tolist()

    subset = df[df["model"].isin(top_models) & df["language"].isin(top_langs)]
    pivot = subset.pivot_table(values="token_count", index="model",
                               columns="language", aggfunc="mean")

    # Reorder
    pivot = pivot.reindex(index=top_models)
    pivot = pivot.reindex(columns=top_langs)

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Avg Token Count")
    ax.set_title("Average Query Length: Model × Language", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "10_model_language_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_dataset_summary(df_raw, df_processed, embeddings, output_dir):
    """Create a summary dashboard with key dataset statistics."""
    fig = plt.figure(figsize=(14, 8))

    # Title
    fig.suptitle("LMSYS-Chat-1M: Dataset Summary Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

    # Stat cards
    stats = [
        ("Raw Queries", f"{len(df_raw):,}", COLORS["primary"]),
        ("Processed", f"{len(df_processed):,}", COLORS["success"]),
        ("Unique Models", f"{df_processed['model'].nunique()}" if "model" in df_processed.columns else "N/A", COLORS["secondary"]),
        ("Languages", f"{df_processed['language'].nunique()}" if "language" in df_processed.columns else "N/A", COLORS["accent"]),
    ]

    for i, (label, value, color) in enumerate(stats):
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.6, value, ha="center", va="center", fontsize=24,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.2, label, ha="center", va="center", fontsize=11,
                color="#667085", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        # Add a subtle bottom border
        ax.axhline(y=0.05, color=color, linewidth=3, xmin=0.2, xmax=0.8)

    # Embedding stats
    embed_stats = [
        ("Embed Dim", f"{embeddings.shape[1]}", COLORS["primary"]),
        ("Embed Size", f"{embeddings.nbytes / 1e6:.1f} MB", COLORS["success"]),
        ("Avg Token Len", f"{df_processed['token_count'].mean():.1f}", COLORS["secondary"]),
        ("Dedup Rate", f"{(1 - len(df_processed)/len(df_raw))*100:.1f}%", COLORS["accent"]),
    ]

    for i, (label, value, color) in enumerate(embed_stats):
        ax = fig.add_subplot(gs[1, i])
        ax.text(0.5, 0.6, value, ha="center", va="center", fontsize=22,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.2, label, ha="center", va="center", fontsize=11,
                color="#667085", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.axhline(y=0.05, color=color, linewidth=3, xmin=0.2, xmax=0.8)

    # Bottom row: mini charts
    ax_len = fig.add_subplot(gs[2, :2])
    ax_len.hist(df_processed["token_count"], bins=50, color=COLORS["primary"], alpha=0.7,
                edgecolor="white", linewidth=0.3)
    ax_len.set_title("Token Length Distribution", fontsize=10, fontweight="bold")
    ax_len.set_xlabel("Tokens", fontsize=9)

    if "model" in df_processed.columns:
        ax_model = fig.add_subplot(gs[2, 2:])
        top5 = df_processed["model"].value_counts().head(5)
        ax_model.barh(range(5), top5.values, color=PALETTE[:5], alpha=0.85)
        ax_model.set_yticks(range(5))
        ax_model.set_yticklabels(top5.index, fontsize=8)
        ax_model.invert_yaxis()
        ax_model.set_title("Top 5 Models", fontsize=10, fontweight="bold")
        ax_model.set_xlabel("Count", fontsize=9)

    path = output_dir / "00_dataset_summary_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────


@click.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/",
              help="Directory containing parquet files.")
@click.option("--embed-dir", type=click.Path(exists=True), default="results/embeddings/",
              help="Directory containing embedding .npy files.")
@click.option("--output", type=click.Path(), default="results/figures/",
              help="Output directory for PNG figures.")
@click.option("--tsne-samples", type=int, default=3000,
              help="Number of samples for t-SNE visualization.")
@click.option("--sim-samples", type=int, default=2000,
              help="Number of samples for similarity distribution.")
def main(data_dir, embed_dir, output, tsne_samples, sim_samples):
    """Generate Phase 1 data and embedding visualizations."""
    set_style()

    data_dir = Path(data_dir)
    embed_dir = Path(embed_dir)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("Loading data files...")
    logger.info("=" * 60)

    raw_path = data_dir / "raw_queries.parquet"
    proc_path = data_dir / "processed_queries.parquet"

    if not raw_path.exists():
        logger.error(f"Raw queries not found: {raw_path}")
        sys.exit(1)
    if not proc_path.exists():
        logger.error(f"Processed queries not found: {proc_path}")
        sys.exit(1)

    df_raw = pd.read_parquet(raw_path)
    df_processed = pd.read_parquet(proc_path)
    logger.info(f"Raw queries: {len(df_raw)} rows")
    logger.info(f"Processed queries: {len(df_processed)} rows")

    # Load embeddings
    embed_path = embed_dir / "full_embeddings.npy"
    if not embed_path.exists():
        logger.error(f"Embeddings not found: {embed_path}")
        sys.exit(1)

    embeddings = np.load(embed_path)
    logger.info(f"Embeddings: {embeddings.shape}")

    # Generate all plots
    logger.info("=" * 60)
    logger.info("Generating visualizations...")
    logger.info("=" * 60)

    plot_dataset_summary(df_raw, df_processed, embeddings, output_dir)
    plot_query_length_distribution(df_raw, df_processed, output_dir)
    plot_duplicate_frequency(df_processed, output_dir)

    if "model" in df_processed.columns:
        plot_lmsys_model_distribution(df_processed, output_dir)
    if "language" in df_processed.columns:
        plot_language_distribution(df_processed, output_dir)
    if "num_turns" in df_processed.columns:
        plot_conversation_turns(df_processed, output_dir)

    plot_preprocessing_funnel(df_raw, df_processed, output_dir)
    plot_embedding_similarity_distribution(embeddings, output_dir, sample_size=sim_samples)
    plot_embedding_tsne(embeddings, df_processed, output_dir, sample_size=tsne_samples)
    plot_embedding_norms(embeddings, output_dir)

    if "model" in df_processed.columns and "language" in df_processed.columns:
        plot_model_query_length_heatmap(df_processed, output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — All visualizations saved!")
    logger.info("=" * 60)

    figures = sorted(output_dir.glob("*.png"))
    for f in figures:
        logger.info(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    logger.info(f"\nTotal: {len(figures)} figures in {output_dir}")


if __name__ == "__main__":
    main()
