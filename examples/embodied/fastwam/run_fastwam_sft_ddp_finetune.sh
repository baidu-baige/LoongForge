#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_fastwam_sft_ddp_finetune.sh - FastWAM SFT Launch Script (DDP)
#
# Usage:
#   bash run_fastwam_sft_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_fastwam_sft_ddp_finetune.sh                          # override via env
#   bash run_fastwam_sft_ddp_finetune.sh --train-iters 50                        # override a training param (flag form)
#   bash run_fastwam_sft_ddp_finetune.sh model.action_dit_pretrained_path=/path   # override YAML model:/data: fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/"}

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29519"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/Wan2.2-TI2V-5B"}
DATASET_PATH=${DATASET_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/fastwam_sft_ddp"}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/text_embeds"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"fastwam"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy fastwam
    --dataset-path "$DATASET_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type libero_franka
    --num-workers "$NUM_WORKERS"
    --lerobotdataset-version v2.1
    --video-backend pyav
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
SEED=${SEED:-3047}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1.0e-8
    --lr-decay-style cosine_warmup_with_min_lr
    --lr-warmup-iters 0
    --min-lr 1.0e-9
    # Optimizer
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
)

if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
    TRAINING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 10
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Model/data dotlist overrides ──────────────────────────────
MODEL_DATA_OVERRIDES=()
if [[ -n "$ACTION_DIT_PRETRAINED_PATH" ]]; then
    MODEL_DATA_OVERRIDES+=("model.action_dit_pretrained_path=$ACTION_DIT_PRETRAINED_PATH")
fi
if [[ -n "$TEXT_EMBEDDING_CACHE_DIR" ]]; then
    MODEL_DATA_OVERRIDES+=("data.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE_DIR")
fi

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (DDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Data:       $DATASET_PATH"
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
    "${MODEL_DATA_OVERRIDES[@]+"${MODEL_DATA_OVERRIDES[@]}"}" \
    "$@"