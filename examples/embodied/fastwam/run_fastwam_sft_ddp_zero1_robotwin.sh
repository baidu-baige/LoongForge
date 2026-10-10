#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_fastwam_sft_ddp_zero1_robotwin.sh
# FastWAM SFT — RoboTwin 2.0, official-aligned (first-frame/uncond, 30k).
#
# 对齐 standalone（FastWAM repo，RoboTwin 复现）：
#   - data      : RoboTwin 2.0 单数据集（3cam 240x320），T 布局（camera_layout=robotwin_t）
#   - 归一化     : z-score（data.norm_stats_path=官方 dataset_stats.json，data.fastwam_norm_mode=z-score）
#   - action    : 14D；proprio 14D
#   - video DiT : 预训练（skip_dit_load_from_pretrain=false）
#   - action DiT: null（随机初始化，照 standalone）
#   - 文本条件   : 预计算 cache（load_text_encoder=false）
#   - lr        : 1e-4，warmup 1500（30k 步 × 5%），min-lr 1e-6（standalone 5% warmup + eta_min=1e-6）
#   - 显存       : per-device-batch 4 + grad-accum 4（GBS 128，同 standalone bs4+ga4）
#   - 步数       : 30000
#
# Usage:
#   bash run_fastwam_sft_ddp_zero1_robotwin.sh [extra args passed to train.py]
#
# Key env vars: LOONGFORGE_PATH, DIFFSYNTH_MODEL_BASE_PATH, TOKENIZER_PATH,
#   DATASET_PATH, TEXT_EMBEDDING_CACHE_DIR, PRETRAINED_NORM_STATS, OUTPUT_DIR,
#   TRAIN_ITERS, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, SEED
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"/path/to/models"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"}
export DATASET_PATH=${DATASET_PATH:-"$DIFFSYNTH_MODEL_BASE_PATH/data/robotwin2.0/robotwin2.0"}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"/path/to/text_embeds_cache/robotwin"}
export PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS:-"$DIFFSYNTH_MODEL_BASE_PATH/data/robotwin2.0/dataset_stats.json"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/fastwam_sft_robotwin"}
export TRAIN_ITERS=${TRAIN_ITERS:-30000}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2500}
export SEED=${SEED:-42}
# RoboTwin standalone 用随机 action expert（action_dit_pretrained_path=null）
export ACTION_DIT_PRETRAINED_PATH=""

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (RoboTwin 2.0, 30k)"
echo "  Data:      $DATASET_PATH"
echo "  Iters:     $TRAIN_ITERS  batch=$PER_DEVICE_BATCH_SIZE ga=$GRADIENT_ACCUMULATION_STEPS"
echo "  Output:    $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

# NOTE: dotlist overrides MUST come before trailing optionals (--lr-base ...);
# argparse's positional `overrides` fails with "unrecognized arguments" if the
# overrides appear after an optional that takes a value.
bash "$LOONGFORGE_PATH/examples/embodied/fastwam/run_fastwam_sft_ddp_zero1_finetune.sh" \
    data.camera_layout=robotwin_t \
    "data.norm_stats_path=$PRETRAINED_NORM_STATS" \
    data.fastwam_norm_mode=z-score \
    model.action_dim=14 \
    model.proprio_dim=14 \
    model.action_dit_pretrained_path=null \
    model.action_scheduler.train_shift=1.0 \
    model.action_scheduler.infer_shift=1.0 \
    model.redirect_common_files=false \
    --lr-base 1e-4 --lr-warmup-iters 1500 --min-lr 1e-6 \
    --save-format dcp --async-save \
    "$@"
