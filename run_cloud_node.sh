#!/bin/bash
set -euo pipefail

: "${NNODES:?Set NNODES to the total node count (for example, 4)}"
: "${NODE_RANK:?Set NODE_RANK to this node's 0-based rank}"
: "${MASTER_ADDR:?Set MASTER_ADDR to node0's private IP}"
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to the private NIC name (for example, eth0)}"

export MASTER_PORT="${MASTER_PORT:-48123}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export RDZV_ID="${RDZV_ID:-pytorch-transformer-3d}"

export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-2}"
export NUM_MICROBATCHES="${NUM_MICROBATCHES:-2}"
export MODEL_FOLDER="${MODEL_FOLDER:-/mnt/training-data/weights}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

case "${MODEL_FOLDER}" in
    /*) ;;
    *)
        echo "MODEL_FOLDER must be an absolute shared path, got: ${MODEL_FOLDER}" >&2
        exit 1
        ;;
esac

echo "=========================================="
echo "Starting cloud 3D training launch"
echo "=========================================="
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "TP_SIZE=${TP_SIZE} PP_SIZE=${PP_SIZE} DP_SIZE=${DP_SIZE} NUM_MICROBATCHES=${NUM_MICROBATCHES}"
echo "MODEL_FOLDER=${MODEL_FOLDER}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_3d.py