#!/bin/bash
#SBATCH --job-name=clever_bench
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.err
#SBATCH --account=stats_dept2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yashkulk@umich.edu

set -e

echo "=========================================="
echo "  CLEVER Index Benchmark — CPU Job"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  CPUs:   $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo "  Time:   $(date)"
echo "=========================================="

# --- Environment Setup ---
module load python/3.11

# Navigate to project root (submit sbatch from CLEVER/ directory)
cd $SLURM_SUBMIT_DIR

source ${PWD}/venv/bin/activate

echo ""
echo "Python: $(python --version)"
echo ""

# --- Step 1: Run Benchmarks at Each Scale (skip if done) ---
for SIZE in 10k 50k 100k 500k; do
    RESULT_FILE="results/benchmarks/index_benchmark_${SIZE}.json"

    if [ -f "$RESULT_FILE" ]; then
        echo "=== Skipping ${SIZE}: ${RESULT_FILE} already exists ==="
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  Benchmarking: ${SIZE}"
    echo "  Time: $(date)"
    echo "=========================================="

    python scripts/03_run_index_benchmark.py \
        --embed-dir results/embeddings/ \
        --size ${SIZE} \
        --output results/benchmarks/ \
        --config configs/index_benchmark.yaml

    echo "  ${SIZE} complete!"
done

# --- Step 2: Generate Visualizations ---
echo ""
echo "=========================================="
echo "  Generating Benchmark Visualizations"
echo "  Time: $(date)"
echo "=========================================="

python scripts/05_visualize_benchmarks.py \
    --results-dir results/benchmarks/ \
    --output results/figures/

echo ""
echo "=========================================="
echo "  All benchmarks complete!"
echo "  Time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -lh results/benchmarks/
echo ""
echo "Figures:"
ls -lh results/figures/
