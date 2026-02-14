#!/bin/bash
#SBATCH --job-name=clever_embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_%j.log
#SBATCH --error=logs/embed_%j.err
#SBATCH --account=stats_dept2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yashkulk@umich.edu

set -e

echo "=========================================="
echo "  CLEVER Embedding Generation — GPU Job"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  GPUs:   $CUDA_VISIBLE_DEVICES"
echo "  Time:   $(date)"
echo "=========================================="

# --- Environment Setup ---
module load python/3.11 cuda/12.1

# Navigate to project root (works regardless of where sbatch is submitted from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

source ${PWD}/venv/bin/activate

echo ""
echo "Python: $(python --version)"
echo "CUDA:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# --- Step 1: Download & Preprocess ---
echo "=== Step 1: Download & Preprocess ==="
if [ ! -f data/processed_queries.parquet ]; then
    python scripts/01_download_dataset.py \
        --output data/ \
        --min-tokens 3 \
        --max-tokens 512
else
    echo "Processed queries already exist, skipping download."
fi

# --- Step 2: Generate Embeddings (GPU) ---
echo ""
echo "=== Step 2: Generate Embeddings (GPU) ==="
python scripts/02_generate_embeddings.py \
    --input data/processed_queries.parquet \
    --output results/embeddings/ \
    --device cuda \
    --batch-size 512 \
    --sizes 10k,50k,100k,500k,full

# --- Step 3: Generate Visualizations ---
echo ""
echo "=== Step 3: Generate Visualizations ==="
python scripts/04_visualize_data.py \
    --data-dir data/ \
    --embed-dir results/embeddings/ \
    --output results/figures/

echo ""
echo "=========================================="
echo "  Pipeline complete! (embed + visualize)"
echo "  Time: $(date)"
echo "=========================================="
