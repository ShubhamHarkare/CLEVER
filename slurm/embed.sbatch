#!/bin/bash
#SBATCH --job-name=clever_embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cse584w25_class
#SBATCH --output=logs/embed_%j.log
#SBATCH --error=logs/embed_%j.err

# ============================================================
#  CLEVER — Embedding Generation on Great Lakes (GPU)
#
#  This job:
#    1. Downloads LMSYS-Chat-1M (if not already cached)
#    2. Preprocesses queries
#    3. Generates embeddings on GPU with all-MiniLM-L6-v2
#    4. Creates all scale subsets (10K–1M)
#
#  Submit with:
#    mkdir -p logs
#    sbatch slurm/embed.sbatch
#
#  NOTE: Confirm your SLURM account name with:
#    sacctmgr show assoc user=$USER
#    Update --account above if different.
# ============================================================

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

# Activate virtual environment
# Update this path to match your Great Lakes setup
source ~/cse584-venv/bin/activate

# Navigate to project root
cd $SLURM_SUBMIT_DIR/..

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

echo ""
echo "=========================================="
echo "  Embedding generation complete!"
echo "  Time: $(date)"
echo "=========================================="
