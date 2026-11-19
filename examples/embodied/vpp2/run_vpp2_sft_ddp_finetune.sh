#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_vpp2_sft_ddp_finetune.sh - VPP2 Action SFT Launch Script (DDP)
#
# Usage:
#   bash run_vpp2_sft_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_vpp2_sft_ddp_finetune.sh           # override via env
#   bash run_vpp2_sft_ddp_finetune.sh --train-iters 500        # override a training param
#   bash run_vpp2_sft_ddp_finetune.sh model.action_horizon=16  # override model:/data: fields
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${LOONGFORGE_PYTHON_BIN:-/usr/bin/python3}
export PYTHONNOUSERSITE=1

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export VPP2_ASSETS_ROOT=${VPP2_ASSETS_ROOT:-"/workspace/vpp2/vpp2_assets"}

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29529"}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Paths ─────────────────────────────────────────────────────
export VPP2_VIDEO_CHECKPOINT=${VPP2_VIDEO_CHECKPOINT:-"$VPP2_ASSETS_ROOT/libero_video_step010000.pt"}
export VPP2_DATASET_ROOT=${VPP2_DATASET_ROOT:-"$VPP2_ASSETS_ROOT/libero_subset"}
DATASET_PATH=${DATASET_PATH:-"$VPP2_DATASET_ROOT/libero_spatial_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_object_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_goal_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_10_no_noops_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vpp2_ddp"}

# ── Model config ──────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name vpp2
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy vpp2
    --dataset-path "$DATASET_PATH"
    --video-backend pyav
)

# ── Training params ───────────────────────────────────────────
# Keep the existing DDP regression recipe.
TRAIN_ITERS=${TRAIN_ITERS:-100}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps 1
    --seed 42
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1.0e-4
    --lr-decay-style linear_warmup_cosine_annealing
    --lr-warmup-iters 500
    --min-lr 1.0e-6
    # Optimizer
    --optimizer AdamW
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 0.0
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --no-check-for-nan-in-loss-and-grad
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --no-ddp-find-unused-parameters
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge VPP2 Action Training (DDP)"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Model:      vpp2"
echo "  Checkpoint: $VPP2_VIDEO_CHECKPOINT"
echo "  Data:       $DATASET_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
    "$PYTHON_BIN" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
