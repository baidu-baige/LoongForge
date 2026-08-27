#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_xvla_ddp_finetune.sh - X-VLA Training Launch Script (DDP)
#
# Usage:
#   bash run_xvla_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_xvla_ddp_finetune.sh                  # override via env
#   bash run_xvla_ddp_finetune.sh --lr-base 1e-4                  # override a training param (flag form)
#   bash run_xvla_ddp_finetune.sh backbone.image_size=448         # override YAML fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29235"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/xvla/models/X-VLA-WidowX"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/xvla/models/X-VLA-WidowX"}
export DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/xvla/datasets/XVLA-Soft-Fold/0928_10am_new"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/xvla_ddp"}

# nsys profiling is OFF by default — wrapping torchrun with `nsys profile` makes
# the run appear to "hang" after the last training step while nsys collects and
# writes the .nsys-rep report (slow for multi-view VLA, and it also waits on
# re-parented child processes). Enable explicitly with XVLA_PROFILE=1.
NSYS_CMD=()
if [ "${XVLA_PROFILE:-0}" = "1" ]; then
    NSYS_CMD=(
        nsys profile
        --output="$OUTPUT_DIR/nsys_report"
        -s none --trace=cuda,nvtx,osrt
        --force-overwrite=true
    )
fi

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"xvla"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-4}
DATA_ARGS=(
    --dataset-format hdf5_datasets
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type libero_franka
    --num-workers "$NUM_WORKERS"
    --batch-drop-last
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
SEED=${SEED:-42}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 0.0001
    --lr-group "model.vlm=1e-5,model.transformer.soft_prompt_hub=1e-5"
    --lr-warmup-iters 5
    --loss-spike-threshold 1000
    # Optimizer
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --pretrained-checkpoint "$CHECKPOINT_PATH"
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge X-VLA Training (DDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    "${NSYS_CMD[@]}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
