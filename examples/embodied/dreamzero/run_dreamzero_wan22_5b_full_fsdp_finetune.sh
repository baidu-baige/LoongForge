#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# DreamZero Wan2.2-5B full fine-tuning with FSDP.
#
# Usage:
#   bash run_dreamzero_wan22_5b_full_fsdp_finetune.sh
#   CACHE_DIR=/path/to/dreamzero_cache \
#     bash run_dreamzero_wan22_5b_full_fsdp_finetune.sh
#   PER_DEVICE_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=32 \
#     bash run_dreamzero_wan22_5b_full_fsdp_finetune.sh
#   bash run_dreamzero_wan22_5b_full_fsdp_finetune.sh --train-iters 100
#
# Optional offline features are enabled by setting CACHE_DIR to an artifact
# produced by precompute_dreamzero_cache.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}

# Distributed launch
GPUS_PER_NODE=${GPUS_PER_NODE:-${NUM_GPUS:-8}}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# Paths
DREAMZERO_DATA_ROOT=${DREAMZERO_DATA_ROOT:-"$LOCAL_VLA_ARTIFACTS_ROOT/dreamzero/datasets"}
DREAMZERO_CKPT_ROOT=${DREAMZERO_CKPT_ROOT:-"$LOCAL_VLA_ARTIFACTS_ROOT/dreamzero/models"}

export WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_CKPT_ROOT/Wan2.1-I2V-14B-480P"}
export WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_CKPT_ROOT/Wan2.2-TI2V-5B"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/dreamzero/tokenizers/umt5-xxl"}
DATA_PATH=${DATA_PATH:-"${DROID_DATA_ROOT:-$DREAMZERO_DATA_ROOT/droid_lerobot}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/dreamzero/droid_wan22_5b_full_fsdp"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}

# Model config
MODEL_NAME=${MODEL_NAME:-"dreamzero_full_wan22_5b"}

# Feature cache
CACHE_ARGS=()
CACHE_DESCRIPTION="disabled (online features)"
CACHE_DIR=${CACHE_DIR:-}
CACHE_MANIFEST=${CACHE_MANIFEST:-}
SAMPLE_TRANSFORM_SEED=${SAMPLE_TRANSFORM_SEED:-0}
if [[ -n "$CACHE_DIR" ]]; then
    CACHE_MANIFEST=${CACHE_MANIFEST:-"$CACHE_DIR/manifest.json"}
    if [[ ! -f "$CACHE_MANIFEST" ]]; then
        echo "DreamZero cache manifest not found: $CACHE_MANIFEST" >&2
        exit 2
    fi
    CACHE_ARGS=(
        model.precomputed_cache.enabled=true
        model.precomputed_cache.cache_dir="$CACHE_DIR"
        model.precomputed_cache.manifest="$CACHE_MANIFEST"
        model.precomputed_cache.strict=true
        model.precomputed_cache.first_frame_only=true
        model.precomputed_cache.validation.validate_artifact=true
        model.precomputed_cache.validation.require_success=true
        model.precomputed_cache.validation.require_transform_config=true
        model.batch_vae_encode=false
        data.use_sample_transform_seed=true
        data.sample_transform_seed="$SAMPLE_TRANSFORM_SEED"
    )
    CACHE_DESCRIPTION="$CACHE_DIR"
fi

# Batch and data pipeline
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((GPUS_PER_NODE * NNODES * PER_DEVICE_BATCH_SIZE))}
LOCAL_BATCH_SIZE=$((GPUS_PER_NODE * NNODES * PER_DEVICE_BATCH_SIZE))
NUM_WORKERS=${NUM_WORKERS:-8}
SAMPLER_NUM_WORKERS=${SAMPLER_NUM_WORKERS:-4}
SAMPLER_WORKER_BATCHING=${SAMPLER_WORKER_BATCHING:-upstream_iterable}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-16}
if (( GLOBAL_BATCH_SIZE % LOCAL_BATCH_SIZE != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by GPUS_PER_NODE * NNODES * PER_DEVICE_BATCH_SIZE" >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE / LOCAL_BATCH_SIZE))}

# Schedule and activation checkpointing
TRAIN_ITERS=${TRAIN_ITERS:-200000}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(((TRAIN_ITERS + 19) / 20))}
OPTIMIZER=${OPTIMIZER:-TorchFusedAdamW}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SEED=${SEED:-42}
LOG_INTERVAL=${LOG_INTERVAL:-1}
ACTIVATION_CHECKPOINT_MODULE_PATTERNS=${ACTIVATION_CHECKPOINT_MODULE_PATTERNS-}
ACTIVATION_CHECKPOINT_SKIP_MODULES=${ACTIVATION_CHECKPOINT_SKIP_MODULES-}
ACTIVATION_CHECKPOINT_ARGS=()
if [[ -n "$ACTIVATION_CHECKPOINT_MODULE_PATTERNS" ]]; then
    ACTIVATION_CHECKPOINT_ARGS+=(
        --activation-checkpoint-module-patterns "$ACTIVATION_CHECKPOINT_MODULE_PATTERNS"
    )
fi
if [[ -n "$ACTIVATION_CHECKPOINT_SKIP_MODULES" ]]; then
    ACTIVATION_CHECKPOINT_ARGS+=(
        --activation-checkpoint-skip-modules "$ACTIVATION_CHECKPOINT_SKIP_MODULES"
    )
fi

# FSDP recipe
FSDP_WRAP_MODULES=${FSDP_WRAP_MODULES-"WanTextEncoder,VisionTransformer,CausalWanAttentionBlock"}
FSDP_FORWARD_PREFETCH_DISTANCE=${FSDP_FORWARD_PREFETCH_DISTANCE:-1}
FSDP_BACKWARD_PREFETCH_DISTANCE=${FSDP_BACKWARD_PREFETCH_DISTANCE:-1}
FSDP_RESHARD_DEFAULT=${FSDP_RESHARD_DEFAULT:-false}
if [[ -n "$CACHE_DIR" ]]; then
    # These modules are only validated for strict precomputed-cache training,
    # where the VAE is not executed during the training step.
    FSDP_IGNORE_FROZEN_MODULE_CLASSES=${FSDP_IGNORE_FROZEN_MODULE_CLASSES-"WanVideoVAE38,VisionTransformer"}
else
    # Keep online feature extraction conservative. It can still be enabled
    # explicitly after validating that workload's numerics and memory budget.
    FSDP_IGNORE_FROZEN_MODULE_CLASSES=${FSDP_IGNORE_FROZEN_MODULE_CLASSES-}
fi
FSDP_IGNORED_FROZEN_PARAM_DTYPE=${FSDP_IGNORED_FROZEN_PARAM_DTYPE:-bf16}
FSDP_DELTA_FP8_ALLGATHER=${FSDP_DELTA_FP8_ALLGATHER:-1}
FSDP_DELTA_FP8_BLOCK=${FSDP_DELTA_FP8_BLOCK:-256}
FSDP_DELTA_FP8_PRIME_STEPS=${FSDP_DELTA_FP8_PRIME_STEPS:-1}
FSDP_DELTA_FP8_REPRIME_INTERVAL=${FSDP_DELTA_FP8_REPRIME_INTERVAL:-0}

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy fsdp
    --dtype bfloat16
    --fsdp-original-param-dtype fp32
    --fsdp-unshard-param-dtype bf16
    --fsdp-reduce-dtype bf16
    --fsdp-cast-forward-inputs
    --fsdp-wrap-modules "$FSDP_WRAP_MODULES"
    --fsdp-forward-prefetch-distance "$FSDP_FORWARD_PREFETCH_DISTANCE"
    --fsdp-backward-prefetch-distance "$FSDP_BACKWARD_PREFETCH_DISTANCE"
    --fsdp-reshard-default "$FSDP_RESHARD_DEFAULT"
)
if [[ -n "$FSDP_IGNORE_FROZEN_MODULE_CLASSES" ]]; then
    DISTRIBUTED_TRAINING_ARGS+=(
        --fsdp-ignore-frozen-module-classes "$FSDP_IGNORE_FROZEN_MODULE_CLASSES"
        --fsdp-ignored-frozen-param-dtype "$FSDP_IGNORED_FROZEN_PARAM_DTYPE"
    )
fi
if [[ "$FSDP_DELTA_FP8_ALLGATHER" == "1" ]]; then
    DISTRIBUTED_TRAINING_ARGS+=(
        --fsdp-delta-fp8-allgather
        --fsdp-delta-fp8-block "$FSDP_DELTA_FP8_BLOCK"
        --fsdp-delta-fp8-prime-steps "$FSDP_DELTA_FP8_PRIME_STEPS"
        --fsdp-delta-fp8-reprime-interval "$FSDP_DELTA_FP8_REPRIME_INTERVAL"
    )
fi

MODEL_CONFIG_ARGS=(--model-name "$MODEL_NAME")
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy dreamzero
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --video-backend decord
    --num-workers "$NUM_WORKERS"
    --dataloader-prefetch-factor "$DATALOADER_PREFETCH_FACTOR"
)
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    --lr-base "${LR:-1.0e-5}"
    --min-lr 0
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --optimizer "$OPTIMIZER"
    --clip-grad 1.0
    --weight-decay 1.0e-5
    --weight-decay-grouping bias_norm
    --adam-beta1 0.95
    --adam-beta2 0.999
    --adam-eps 1.0e-8
    --save-interval "$SAVE_INTERVAL"
    --save-format dcp
    "${ACTIVATION_CHECKPOINT_ARGS[@]}"
)
LOGGING_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --wandb-mode "${WANDB_MODE:-disabled}"
    --wandb-project "${WANDB_PROJECT:-dreamzero}"
)
MODEL_DATA_OVERRIDES=(
    data.sampler_num_workers="$SAMPLER_NUM_WORKERS"
    data.sampler_worker_batching="$SAMPLER_WORKER_BATCHING"
)

echo "========================================================================"
echo "  LoongForge DreamZero Wan2.2-5B Full Fine-Tuning (FSDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Data:       $DATA_PATH"
echo "  Wan2.2:     $WAN22_CKPT_DIR"
echo "  Wan2.1:     $WAN21_CKPT_DIR"
echo "  Cache:      $CACHE_DESCRIPTION"
echo "  Output:     $OUTPUT_DIR"
echo "========================================================================"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@" \
    "${CACHE_ARGS[@]}" \
    "${MODEL_DATA_OVERRIDES[@]}"
