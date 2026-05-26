#!/bin/bash
# WSL2 / Linux local 1-GPU smoke test
# Configured to fit on a 4GB VRAM GPU without modifying config.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 1x1x1 topology (1 rank total)
export TP_SIZE=1
export PP_SIZE=1
export DP_SIZE=1

# Small batch size to avoid Out-Of-Memory (OOM) on 4GB VRAM
export BATCH_SIZE=1
export NUM_MICROBATCHES=1
export NUM_EPOCHS=1

# Store checkpoints locally in a temporary folder
export MODEL_FOLDER="./wsl_smoke_weights"

# WSL2 sometimes struggles with NCCL initialization depending on the Windows build.
# For a 1-GPU smoke test, Gloo is perfectly safe and bypassing NCCL avoids hangs.
export DIST_BACKEND="gloo"

echo "=========================================="
echo "Starting 1-GPU Smoke Test on Linux/WSL..."
echo "=========================================="

torchrun --standalone --nproc_per_node=1 train_3d.py