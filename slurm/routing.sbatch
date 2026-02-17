#!/bin/bash
#SBATCH --job-name=clever-routing
#SBATCH --account=stats_dept2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/routing_%j.log
#SBATCH --error=logs/routing_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ──────────────────────────────────────────────────────────────
# Phase 3: Cost-Based Query Routing Evaluation
# Runs the routing evaluation pipeline on Great Lakes.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

echo "=== CLEVER Phase 3: Routing Evaluation ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   $SLURM_MEM_PER_NODE MB"
echo "Started:  $(date)"
echo ""

# ── Setup ────────────────────────────────────────────────────
cd ~/scratch/yashkulk/CSE584/CLEVER

# Activate virtual environment
source venv/bin/activate

# Create log and output directories
mkdir -p logs results/routing results/figures

# ── Run routing evaluation ────────────────────────────────────
echo ">>> Running routing evaluation..."
python scripts/06_run_routing_eval.py \
    --embeddings results/embeddings/full_embeddings.npy \
    --queries data/full_queries.parquet \
    --config configs/routing.yaml \
    --output results/routing/

echo ""
echo ">>> Routing evaluation complete."

# ── Generate visualizations ────────────────────────────────────
echo ">>> Generating routing visualizations..."
python scripts/07_visualize_routing.py \
    --results-dir results/routing/ \
    --output results/figures/

echo ""
echo "=== Phase 3 Complete ==="
echo "Finished: $(date)"
echo "Results:  results/routing/"
echo "Figures:  results/figures/"
