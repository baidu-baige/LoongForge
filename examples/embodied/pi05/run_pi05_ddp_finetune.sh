#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_pi05_ddp_finetune.sh - pi05 VLA SFT Launch Script (DDP)
#
# Usage:
#   bash run_pi05_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_pi05_ddp_finetune.sh                          # override via env
#   bash run_pi05_ddp_finetune.sh --train-iters 50                        # override a training param (flag form)
#   bash run_pi05_ddp_finetune.sh model.state_dim=8 data.image_size=448   # override YAML model:/data: fields (dotlist form)
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

export NO_ALBUMENTATIONS_UPDATE=1

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/pi05/tokenizers/paligemma-3b-pt-224"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/pi05/models/pi05_base"}
DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/pi05/datasets/libero"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/pi05_ddp"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"pi05"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type libero_franka
    --batch-drop-last
    --num-workers "$NUM_WORKERS"
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-12}
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
    --lr-base 2.5e-5
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 10
    --min-lr 1.0e-6
    # Optimizer
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --pretrained-checkpoint "$CHECKPOINT_PATH"
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --ddp-find-unused-parameters
    --ddp-static-graph
    --ddp-skip-all-reduce-unused-params
    --ddp-gradient-as-bucket-view
    --no-dynamo-optimize-ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "$TENSORBOARD_DIR"
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge pi05 Training (DDP)"
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
