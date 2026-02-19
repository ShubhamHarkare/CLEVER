#!/bin/bash
#SBATCH --job-name=eviction_eval
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cse584w25_class
#SBATCH --output=logs/eviction_%j.log
#SBATCH --error=logs/eviction_%j.err
#SBATCH --mail-type=END,FAIL

# ── Environment ──────────────────────────────────────────────────
module load python/3.11
source ~/cse584-venv/bin/activate

cd ~/CLEVER

echo "═══════════════════════════════════════════════════════════"
echo "Phase 4: Eviction Policy Evaluation"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════"

mkdir -p results/eviction results/figures logs

# ── Run eviction experiments (multi-seed) ────────────────────────
python scripts/08_run_eviction.py \
    --embeddings results/embeddings/full_embeddings.npy \
    --queries data/full_queries.parquet \
    --config configs/eviction.yaml \
    --output results/eviction/ \
    --multi-seed

echo ""
echo "Eviction experiment completed: $(date)"
echo ""

# ── Generate figures ─────────────────────────────────────────────
python scripts/09_visualize_eviction.py \
    --results results/eviction/eviction_results_multi_seed.json \
    --output results/figures/

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Phase 4 COMPLETE: $(date)"
echo "═══════════════════════════════════════════════════════════"
