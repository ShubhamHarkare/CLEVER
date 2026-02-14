#!/usr/bin/env bash
# 00_setup_environment.sh — Verify that the environment is correctly set up.
#
# Usage:
#   bash scripts/00_setup_environment.sh

set -e

echo "============================================"
echo "  CLEVER — Environment Setup Verification"
echo "============================================"
echo ""

# --- Check Python version ---
echo "[1/4] Checking Python version..."
python3 -c "
import sys
v = sys.version_info
print(f'  Python {v.major}.{v.minor}.{v.micro}')
assert v >= (3, 11), f'Need Python 3.11+, got {v.major}.{v.minor}'
print('  ✓ Python version OK')
"

# --- Check FAISS ---
echo ""
echo "[2/4] Checking FAISS installation..."
python3 -c "
import numpy as np
import faiss

print(f'  FAISS version: {faiss.__version__ if hasattr(faiss, \"__version__\") else \"unknown\"}')

# Quick sanity test: build a small Flat index and search
d = 384
xb = np.random.rand(1000, d).astype('float32')
index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xb[:5], k=10)
assert I.shape == (5, 10), f'Expected (5, 10), got {I.shape}'

# Verify self-search: closest neighbor should be itself
for i in range(5):
    assert I[i, 0] == i, f'Self-search failed for query {i}'

print('  ✓ FAISS OK — index build + search working')
"

# --- Check sentence-transformers ---
echo ""
echo "[3/4] Checking sentence-transformers..."
python3 -c "
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(['test query for CLEVER benchmark'])
assert emb.shape == (1, 384), f'Expected (1, 384), got {emb.shape}'

# Verify normalization works
emb_norm = model.encode(['test query'], normalize_embeddings=True)
norm = np.linalg.norm(emb_norm[0])
assert abs(norm - 1.0) < 1e-5, f'Normalization failed: norm={norm}'

print(f'  Model: all-MiniLM-L6-v2, dim={emb.shape[1]}')
print('  ✓ Encoder OK — encoding + normalization working')
"

# --- Check other core deps ---
echo ""
echo "[4/4] Checking other dependencies..."
python3 -c "
imports = [
    ('pandas', 'pd'),
    ('numpy', 'np'),
    ('scipy', 'scipy'),
    ('sklearn', 'sklearn'),
    ('yaml', 'yaml'),
    ('click', 'click'),
    ('tqdm', 'tqdm'),
    ('psutil', 'psutil'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'sns'),
]
failed = []
for module, alias in imports:
    try:
        __import__(module)
        print(f'  ✓ {module}')
    except ImportError:
        print(f'  ✗ {module} — NOT FOUND')
        failed.append(module)

if failed:
    print(f'\n  WARNING: Missing packages: {failed}')
    print('  Run: pip install -r requirements.txt')
else:
    print('\n  ✓ All dependencies OK')
"

echo ""
echo "============================================"
echo "  Environment verification complete!"
echo "============================================"
