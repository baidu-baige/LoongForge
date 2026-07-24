#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# LingBot-VA post-training.
# Set all data and checkpoint paths to existing local files; this script runs offline.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOONGFORGE_PATH=${LOONGFORGE_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
MODEL_NAME=${MODEL_NAME:?Dataset launcher must set MODEL_NAME}

CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a local LingBot-VA checkpoint}
DATA_PATH=${DATA_PATH:?Set DATA_PATH to a local LeRobot dataset}
EMPTY_EMB_PATH=${EMPTY_EMB_PATH:-$DATA_PATH/empty_emb.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$LOONGFORGE_PATH/outputs/lingbot_va}

: "${GRADIENT_ACCUMULATION_STEPS:?Dataset launcher must set GRADIENT_ACCUMULATION_STEPS}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
NUM_WORKERS=${NUM_WORKERS:-16}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# User-selectable LingBot features.
export LINGBOT_SAMPLE_META_EXPORT=${LINGBOT_SAMPLE_META_EXPORT:-0}       # Export per-sample metadata for diagnosis.
export LINGBOT_SKIP_FINAL_CHECKPOINT=${LINGBOT_SKIP_FINAL_CHECKPOINT:-0} # Skip only the final checkpoint save.
export LINGBOT_BASELINE_LOSS_LOG=${LINGBOT_BASELINE_LOSS_LOG:-1}         # Emit baseline-compatible loss lines.
export LINGBOT_BALANCED_SAMPLER=${LINGBOT_BALANCED_SAMPLER:-1} # Balance variable-shape samples across ranks.
export LINGBOT_REPO_DISCOVERY_CACHE=${LINGBOT_REPO_DISCOVERY_CACHE:-1} # Cache the baseline-compatible repo discovery order.
export LINGBOT_LAYERWISE_COMPILE=${LINGBOT_LAYERWISE_COMPILE:-1}         # Compile layerwise norm/residual kernels.
export LINGBOT_SAMPLE_META_EXPORT_DIR=${LINGBOT_SAMPLE_META_EXPORT_DIR:-$OUTPUT_DIR/sample_meta}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

TRAINING_ARGS=(
    --model-name "$MODEL_NAME"
    --trainer-type LingBotFinetuneTrainer
    --dataset-format lerobot_datasets
    --dataset-strategy lingbot_va
    --pretrained-checkpoint "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --train-iters "$TRAIN_ITERS"
    --save-interval 1000
    --seed 42
    --batch-drop-last false
    --dataloader-seed-workers
    --dataloader-multiprocessing-context fork
    --num-workers "$NUM_WORKERS"
    --per-device-batch-size 1
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --lr-base 1e-5
    --min-lr 0.0
    --lr-warmup-iters 10
    --lr-decay-style constant_with_warmup
    --optimizer TorchFusedAdamW
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1
    --clip-grad 2.0
    --dtype bfloat16
    --distributed-strategy fsdp
    --fsdp-wrap-modules WanTransformerBlock
    --fsdp-min-num-params 1000000000000000
    --fsdp-leftover-min-num-params 1000000000000000
    --fsdp-reshard-default none
    --fsdp-reshard-root true
    --save-format dcp
    --no-async-save
    --log-interval 1
    --loss-log-rank -1
)

MODEL_DATA_OVERRIDES=(
    "data.dataset_path=$DATA_PATH"
    "data.empty_emb_path=$EMPTY_EMB_PATH"
    model.num_layers=30
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers=30
    model.lingbot_va_use_flex_attention=true
)

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" torchrun "${DISTRIBUTED_ARGS[@]}" "$LOONGFORGE_PATH/loongforge/embodied/train.py" "${TRAINING_ARGS[@]}" "${MODEL_DATA_OVERRIDES[@]}" "$@"
