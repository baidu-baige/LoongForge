#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_groot_n1_7_ddp_finetune.sh - GR00T-N1.7 Training Launch Script (DDP)
#
# Usage:
#   bash run_groot_n1_7_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_groot_n1_7_ddp_finetune.sh            # override via env
#   bash run_groot_n1_7_ddp_finetune.sh --train-iters 500         # override a training param (flag form)
#   bash run_groot_n1_7_ddp_finetune.sh model.action_horizon=32   # override YAML model:/data: fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}

GROOT_N1_7_OPS_SRC=${GROOT_N1_7_OPS_SRC:-"${LOONGFORGE_PATH}/ops/cuda_source/groot_n1_7_op"}

build_groot_n1_7_ops() {
    if python -c "import groot_n1_7_op" > /dev/null 2>&1; then
        echo "GR00T-N1.7 CUDA operators already installed: groot_n1_7_op"
        return
    fi

    if [[ ! -f "${GROOT_N1_7_OPS_SRC}/setup.py" ]]; then
        echo "groot_n1_7_op is not installed and its sources were not found at ${GROOT_N1_7_OPS_SRC}." >&2
        echo "Set GROOT_N1_7_OPS_SRC to the groot_n1_7_op directory and rerun." >&2
        exit 1
    fi

    echo "Installing GR00T-N1.7 CUDA operators from ${GROOT_N1_7_OPS_SRC}"
    NVCC_THREADS=${NVCC_THREADS:-${MAX_JOBS:-8}} \
        pip install --no-build-isolation -e "${GROOT_N1_7_OPS_SRC}"
}

build_groot_n1_7_ops

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

export COSMOS_LOCAL_PATH="${COSMOS_LOCAL_PATH:-$LOCAL_VLA_ARTIFACTS_ROOT/groot_n1_7/models/Cosmos-Reason2-2B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$COSMOS_LOCAL_PATH}"

# ── Paths ─────────────────────────────────────────────────────
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/groot_n1_7/models/GR00T-N1.7-3B"}
DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/groot_n1_7/datasets/cube_to_bowl_5"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/groot_n1_7"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"groot_n1_7"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-4}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy groot_n1_7
    --dataset-path "$DATA_PATH"
    --lerobotdataset-version v2.1
    --video-backend torchcodec
    --num-workers "$NUM_WORKERS"
    --dataloader-prefetch-factor 8
    --dataloader-multiprocessing-context fork
    --distributed-sampler-mode block
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-104}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
SEED=${SEED:-42}

TRAINING_ARGS=(
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1.0e-4
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 5
    --min-lr 0.0
    #--optimizer AdamW
    --optimizer TEFusedAdamW
    --clip-grad 1.0
    --weight-decay 1.0e-5
    --weight-decay-grouping bias_norm
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --pretrained-checkpoint "$CHECKPOINT_PATH"
    --deterministic-mode
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --cuda-graph-warmup-steps 3
    --cuda-graph-pad-length 0
    --cuda-graph-ddp-sync-in-graph
    --no-check-for-nan-in-loss-and-grad
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --ddp-bucket-cap-mb 100
    --no-ddp-find-unused-parameters
    --ddp-static-graph
    # --dataloader-seed-workers
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --loss-log-rank -1
    --wandb-mode disabled
    --tensorboard-dir "$TENSORBOARD_DIR"
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge GR00T-N1.7 Training (DDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
