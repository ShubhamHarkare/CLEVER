#!/bin/bash
#SBATCH --job-name=clever_bench
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=cse584w25_class
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.err

# ============================================================
#  CLEVER — Index Benchmark on Great Lakes (CPU)
#
#  Placeholder for Phase 2 benchmarks.
#  This will run all index comparisons at scale (100K–1M).
#
#  Submit with:
#    mkdir -p logs
#    sbatch slurm/benchmark.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  CLEVER Index Benchmark — CPU Job"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  CPUs:   $SLURM_CPUS_PER_TASK"
echo "  Time:   $(date)"
echo "=========================================="

module load python/3.11
source ~/cse584-venv/bin/activate
cd $SLURM_SUBMIT_DIR/..

echo ""
echo "=== Phase 2: Index Benchmarks ==="
echo "TODO: Implement in Phase 2"
echo ""

# Placeholder command:
# python scripts/03_run_index_benchmark.py \
#     --config configs/index_benchmark.yaml \
#     --embeddings-dir results/embeddings/ \
#     --output results/benchmarks/ \
#     --sizes 100k,500k,full

echo "=========================================="
echo "  Benchmark complete!"
echo "  Time: $(date)"
echo "=========================================="
