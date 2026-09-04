#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_lingbot_va_libero_fsdp_finetune.sh - LingBot-VA LIBERO-Long post-training
# with baseline-aligned data semantics.
#
# Set all data and checkpoint paths to existing local files; this script runs offline.
#
# Usage:
#   bash run_lingbot_va_libero_fsdp_finetune.sh
#   TRAIN_ITERS=50 bash run_lingbot_va_libero_fsdp_finetune.sh   # override via env
#   bash run_lingbot_va_libero_fsdp_finetune.sh --train-iters 50 # override a training param (flag form)
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

# ── Paths ─────────────────────────────────────────────────────
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/lingbot_va/models/lingbot-va-posttrain-libero-long"}
DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/lingbot_va/datasets/libero-long-lerobot"}
EMPTY_EMB_PATH=${EMPTY_EMB_PATH:-"$DATA_PATH/empty_emb.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/lingbot_va_libero"}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# User-selectable LingBot features.
export LINGBOT_BALANCED_SAMPLER=${LINGBOT_BALANCED_SAMPLER:-1}           # Balance variable-shape samples across ranks.
export LINGBOT_FSDP_RESHARD=${LINGBOT_FSDP_RESHARD:-0}                   # 1 restores the framework-default FSDP reshard.
export LINGBOT_FSDP_BF16_REDUCE=${LINGBOT_FSDP_BF16_REDUCE:-1}           # 0 reduces gradients in FP32 instead of BF16.
# Export the per-rank sample loading order for reproducibility checks.
export LINGBOT_SAMPLE_ORDER_EXPORT_DIR=${LINGBOT_SAMPLE_ORDER_EXPORT_DIR:-}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"lingbot_va_libero"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy lingbot_va
    --dataloader-seed-workers
    --dataloader-multiprocessing-context fork
    --num-workers "$NUM_WORKERS"
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
SEED=${SEED:-42}

TRAINING_ARGS=(
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1e-5
    --min-lr 0.0
    --lr-warmup-iters 10
    --lr-decay-style constant_with_warmup
    # Optimizer
    --optimizer TorchFusedAdamW
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1
    --clip-grad 2.0
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --save-format dcp
    --no-async-save
    --pretrained-checkpoint "$CHECKPOINT_PATH"
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy fsdp
    --dtype bfloat16
    --fsdp-wrap-modules WanTransformerBlock
    --fsdp-reshard-default none
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
)

# ── Model/data dotlist overrides ──────────────────────────────
MODEL_DATA_OVERRIDES=(
    "data.dataset_path=$DATA_PATH"
    "data.empty_emb_path=$EMPTY_EMB_PATH"
    model.num_layers=30
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers=30
    model.lingbot_va_use_flex_attention=true
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge LingBot-VA LIBERO-Long Post-Training (FSDP)"
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
    "${MODEL_DATA_OVERRIDES[@]}" \
    "$@"
