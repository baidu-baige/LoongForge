#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# 8-GPU LoongForge-VLDA LIBERO Stage 1 launcher.
#   global batch 96 (12 per GPU), 10k steps, 500 warmup, save every 1k,
#   peak LR 7e-5, cosine decay to 3e-6, gradient checkpointing enabled.
# Stage 1 freezes PaliGemma and the action expert and trains the world model only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PATH="$LOONGFORGE_PATH/.venv/bin:$PATH"

export STAGE=1
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# 8-GPU training recipe.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-96}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export ZERO_OPTIMIZER=${ZERO_OPTIMIZER:-0}
export TRAIN_ITERS=${TRAIN_ITERS:-10000}
export WARMUP_ITERS=${WARMUP_ITERS:-500}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export PEAK_LR=${PEAK_LR:-7e-5}
export MIN_LR=${MIN_LR:-3e-6}
export LOG_INTERVAL=${LOG_INTERVAL:-100}

# Data / model assets (override as needed).
export DATA_PATH=${DATA_PATH:-"$LOONGFORGE_PATH/datasets/libero"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOONGFORGE_PATH/models/paligemma_tokenizer.model"}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOONGFORGE_PATH/models/pi05_base_pytorch"}
export VJEPA_CHECKPOINT=${VJEPA_CHECKPOINT:-"$LOONGFORGE_PATH/models/vjepa2/vitg.pt"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vlda/libero_stage1_8gpu_$(date +%Y%m%d_%H%M%S)"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/loongforge-vlda-inductor}

# Teacher EMA momentum shared by both stages: 0.99925^2 = 0.9985005625.
EMA_MOMENTUM=${EMA_MOMENTUM:-0.9985005625}

exec bash "$SCRIPT_DIR/run_vlda_ddp_finetune.sh" \
    --loss-log-rank=-1 \
    model.gradient_checkpointing=true \
    "model.ema_start=${EMA_MOMENTUM}" \
    "model.ema_end=${EMA_MOMENTUM}" \
    "data.norm_stats_path=${NORM_STATS_PATH:-$DATA_PATH/norm_stats.json}" \
    "$@"
