#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="${LOONGFORGE_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29531}
export VPP2_ASSETS_ROOT=${VPP2_ASSETS_ROOT:-$LOONGFORGE_PATH/vpp2_assets}
export VPP2_VIDEO_CHECKPOINT=${VPP2_VIDEO_CHECKPOINT:-$VPP2_ASSETS_ROOT/libero_video_step010000.pt}
export VPP2_DATASET_ROOT=${VPP2_DATASET_ROOT:-$VPP2_ASSETS_ROOT/libero_subset}
DATASET_PATH=${DATASET_PATH:-$VPP2_DATASET_ROOT/libero_spatial_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_object_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_goal_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_10_no_noops_lerobot}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vpp2_deepspeed_zero2"}
TRAIN_ITERS=${TRAIN_ITERS:-100}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
NUM_WORKERS=${NUM_WORKERS:-16}
# Preserve the reference's complete per-device batch and RNG draw order.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-$LOONGFORGE_PATH/examples/embodied/vpp2/zero2_bf16.json}

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" torchrun \
  --nproc_per_node "$GPUS_PER_NODE" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
  --model-name vpp2 --dataset-format lerobot_datasets --dataset-strategy vpp2 \
  --dataset-path "$DATASET_PATH" --video-backend pyav --num-workers "$NUM_WORKERS" \
  --trainer-type FinetuneTrainer --train-iters "$TRAIN_ITERS" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --seed 42 --set-seed-by-rank --output-dir "$OUTPUT_DIR" \
  --lr-base 1.0e-4 --lr-decay-style linear_warmup_cosine_annealing --lr-warmup-iters 500 \
  --lr-decay-iters 30000 --min-lr 1.0e-6 --optimizer AdamW --weight-decay 0.01 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 0.0 --dtype bfloat16 \
  --save-interval "$SAVE_INTERVAL" --save-format pt --distributed-strategy deepspeed \
  --deepspeed-config "$DEEPSPEED_CONFIG" --no-init-on-meta \
  --no-check-for-nan-in-loss-and-grad --wandb-mode disabled model.disable_train_autocast=true "$@"
