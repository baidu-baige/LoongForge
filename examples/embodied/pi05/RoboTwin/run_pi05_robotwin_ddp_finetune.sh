#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_pi05_robotwin_ddp_finetune.sh - pi05 RoboTwin-2.0 VLA SFT Launch Script (DDP)
#
# Note:
#   DATA_PATH must point to the pi-space dataset view (built via
#   examples/embodied/pi05/build_pidelta_stats.py) whose meta/stats.json holds
#   pi-space action and observation.state statistics.
#
# Usage:
#   bash run_pi05_robotwin_ddp_finetune.sh
#   TRAIN_ITERS=100000 bash run_pi05_robotwin_ddp_finetune.sh
#   bash run_pi05_robotwin_ddp_finetune.sh --resume
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}

# ── Distributed ───────────────────────────────────────────────
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
DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/pi05/datasets/robotwin_unified_pidelta"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/pi05_robotwin"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

# ── Model & Data config ───────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"pi05"}
ACTION_HORIZON=${ACTION_HORIZON:-30}
ACTION_DIM=${ACTION_DIM:-14}
STATE_DIM=${STATE_DIM:-14}
NUM_IMAGES=${NUM_IMAGES:-3}
NORMALIZATION_MODE=${NORMALIZATION_MODE:-"q99"}
USE_ALOHA_PI_DELTA=${USE_ALOHA_PI_DELTA:-"true"}

MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

DOTLIST_OVERRIDES=(
    "model.action_horizon=${ACTION_HORIZON}"
    "model.action_dim=${ACTION_DIM}"
    "model.state_dim=${STATE_DIM}"
    "data.num_images=${NUM_IMAGES}"
    "data.normalization_mode=${NORMALIZATION_MODE}"
    "data.use_aloha_pi_delta_actions=${USE_ALOHA_PI_DELTA}"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type aloha_bimanual
    --batch-drop-last
    --num-workers "$NUM_WORKERS"
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-200000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
LR_BASE=${LR_BASE:-5e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2000}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-200000}
MIN_LR=${MIN_LR:-5e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
SEED=${SEED:-1000}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base "$LR_BASE"
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --min-lr "$MIN_LR"
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
    --activation-checkpoint-module-patterns "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*"
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --ddp-find-unused-parameters
    --ddp-static-graph
    --ddp-gradient-as-bucket-view
    --no-dynamo-optimize-ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 10
    --tensorboard-dir "$TENSORBOARD_DIR"
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge pi05 RoboTwin-2.0 Training (DDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME (horizon=$ACTION_HORIZON, action_dim=$ACTION_DIM, state_dim=$STATE_DIM)"
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
    "$@" \
    "${DOTLIST_OVERRIDES[@]}"
