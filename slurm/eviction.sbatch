#!/bin/bash
#SBATCH --job-name=clever-eviction
#SBATCH --account=stats_dept2
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=eviction_%j.log
#SBATCH --error=eviction_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 4: Semantic-Aware Eviction Policy Evaluation (Multi-Seed)
# Runs the eviction evaluation pipeline on Great Lakes.
#
# NOTE: We use --workers 1 (sequential) because CUDA contexts do
#       not survive fork(). With ProcessPoolExecutor, child processes
#       silently fall back to CPU, making semantic policy ~100× slower.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== CLEVER Phase 4: Eviction Policy Evaluation (Multi-Seed) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   $SLURM_MEM_PER_NODE MB"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started:  $(date)"
echo ""

# ── Setup ────────────────────────────────────────────────────
cd ~/scratch/yashkulk/CSE584/CLEVER

# Activate virtual environment
source venv/bin/activate

# Create log and output directories
mkdir -p logs results/eviction results/figures

# ── Run eviction experiments (multi-seed, sequential) ────────
echo ">>> Running multi-seed eviction evaluation..."
python scripts/08_run_eviction.py \
    --embeddings results/embeddings/full_embeddings.npy \
    --queries data/full_queries.parquet \
    --config configs/eviction.yaml \
    --output results/eviction/ \
    --multi-seed \
    --workers 1

echo ""
echo ">>> Eviction evaluation complete."

# ── Generate visualizations ────────────────────────────────────
echo ">>> Generating eviction visualizations..."
python scripts/09_visualize_eviction.py \
    --results results/eviction/eviction_results_multi_seed.json \
    --output results/figures/

echo ""
echo "=== Phase 4 Complete ==="
echo "Finished: $(date)"
echo "Results:  results/eviction/"
echo "Figures:  results/figures/"
