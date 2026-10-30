#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# 8-GPU LoongForge-VLDA LIBERO Stage 2 launcher.
#   global batch 128 (16 per GPU), 60k steps, save every 5k,
#   warm up to 5e-5 over 10k steps, hold through 40k, cosine decay to 1e-5
#   over the final 20k steps, gradient checkpointing enabled.
# Stage 2 is the real LIBERO action fine-tuning; initialize it from a
# LoongForge Stage 1 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PATH="$LOONGFORGE_PATH/.venv/bin:$PATH"

export STAGE=2
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# 8-GPU training recipe.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export ZERO_OPTIMIZER=${ZERO_OPTIMIZER:-0}
export TRAIN_ITERS=${TRAIN_ITERS:-60000}
export WARMUP_ITERS=${WARMUP_ITERS:-10000}
export LR_DECAY_START_ITERS=${LR_DECAY_START_ITERS:-40000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
export PEAK_LR=${PEAK_LR:-5e-5}
export MIN_LR=${MIN_LR:-1e-5}
export LOG_INTERVAL=${LOG_INTERVAL:-100}

# Data / model assets (override as needed).
export DATA_PATH=${DATA_PATH:-"$LOONGFORGE_PATH/datasets/libero"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOONGFORGE_PATH/models/paligemma_tokenizer.model"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vlda/libero_stage2_8gpu_$(date +%Y%m%d_%H%M%S)"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/loongforge-vlda-inductor}

# CHECKPOINT_PATH must point to a LoongForge Stage 1 checkpoint directory or
# weight file. To resume an interrupted Stage 2 run instead, set RESUME=1 and
# reuse its OUTPUT_DIR (CHECKPOINT_PATH is then ignored by the driver).
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-}

# Teacher EMA momentum: matches the Stage 1 value (0.99925^2 = 0.9985005625).
EMA_MOMENTUM=${EMA_MOMENTUM:-0.9985005625}

exec bash "$SCRIPT_DIR/run_vlda_ddp_finetune.sh" \
    --loss-log-rank=-1 \
    model.gradient_checkpointing=true \
    "model.ema_start=${EMA_MOMENTUM}" \
    "model.ema_end=${EMA_MOMENTUM}" \
    "data.norm_stats_path=${NORM_STATS_PATH:-$DATA_PATH/norm_stats.json}" \
    "$@"
