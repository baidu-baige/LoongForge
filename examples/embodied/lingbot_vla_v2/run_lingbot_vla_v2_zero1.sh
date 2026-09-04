#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# run_lingbot_vla_v2_zero1.sh - lingbot_vla_v2 ZeRO-1 Launch Script (Embodied)
#
# Replicated-compute ZeRO-1: compute parameters are fully replicated, optimizer
# master/state are owned per rank, gradients are reduced to owners and updated
# parameters are synced back. This is NOT PyTorch FSDP1 FULL_SHARD (that path
# is incompatible with Muon's complete-matrix Newton-Schulz).
#
# Defaults reproduce the validated best configuration: 8 GPUs, micro batch 10,
# GBS 80, FP32 gradient reduction, gradient/parameter sync overlap enabled.
#
# Usage:
#   bash run_lingbot_vla_v2_zero1.sh                                  # GBS80 best config, 20 iters
#   bash run_lingbot_vla_v2_zero1.sh > run.log 2>&1                   # capture output (caller's choice)
#   MICRO_BS=16 bash run_lingbot_vla_v2_zero1.sh                      # GBS128, GA1
#   MICRO_BS=8 GRAD_ACCUM=2 bash run_lingbot_vla_v2_zero1.sh          # GBS128, GA2
#   bash run_lingbot_vla_v2_zero1.sh --train-iters 50000              # override a training flag
#   bash run_lingbot_vla_v2_zero1.sh model.gradient_checkpointing=true # override a YAML field
#
# Note: the gradient-ready order is measured inside the run -- iteration 1 uses
# reverse parameter-registration order and iteration 2 onward uses the measured
# order. No order file and no warm-up run is involved.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export AIAK_PATH=${AIAK_PATH:-"$PROJECT_ROOT"}

# -- Paths --------------------------------------------------------------------
DATA_PATH=${DATA_PATH:-"$PROJECT_ROOT/../lingbot-vla-v2/assets/training_data/robotwin.txt"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$PROJECT_ROOT/../dataset/lingbot-vla/Qwen3-VL-4B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/../outputs/lingbot_vla_v2_zero1"}
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# albumentations checks its own version over the network on import.
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

# -- Distributed --------------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29611"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# -- ZeRO-1 knobs -------------------------------------------------------------
# Collective precision and overlap live in the model YAML (model.grad_reduce_dtype,
# model.param_sync_dtype, model.grad_overlap, model.param_overlap,
# model.comm_bucket_mb, model.grad_inflight_mb). Override per run on the command
# line:
#   bash run_lingbot_vla_v2_zero1.sh model.grad_reduce_dtype=fp32
# The precision settings exempt the parameters the model policy marks
# precision-critical (MoE router/gate weights, 1-D norms/biases): expert selection
# is decided by fp32 gate logits, so a rounded router weight flips top-k rather
# than perturbing it.

# -- Model config -------------------------------------------------------------
MODEL_NAME=${MODEL_NAME:-"lingbot_vla_v2"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy lingbot_vla_v2
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --num-workers $NUM_WORKERS
)
if (( NUM_WORKERS > 0 )); then
    DATA_ARGS+=(--dataloader-multiprocessing-context spawn)
fi

# -- Training params ----------------------------------------------------------
# micro batch 10 x 8 GPUs = GBS 80 (micro batch 16 => GBS 128, the GBS ceiling).
MICRO_BS=${MICRO_BS:-10}
GRAD_ACCUM=${GRAD_ACCUM:-1}
USE_COMPILE=${USE_COMPILE:-false}
TRAINING_ARGS=(
    --train-iters ${TRAIN_ITERS:-20}
    --per-device-batch-size $MICRO_BS
    --gradient-accumulation-steps $GRAD_ACCUM
    --seed 42
    --output-dir "$OUTPUT_DIR"
    --manual-gc
    --manual-gc-interval 0
    # Learning rate
    --lr-base 1e-4
    --lr-decay-style constant
    # Optimizer
    --optimizer Muon
    --clip-grad 1
    --weight-decay 0
    --save-interval ${SAVE_INTERVAL:-1000}
    --no-save-training-state
)
if [[ ${USE_NSYS_PROFILE:-0} == 1 ]]; then
    export NSYS_GPU_METRICS_DEVICE=${NSYS_GPU_METRICS_DEVICE:-all}
    TRAINING_ARGS+=(
        --use-nsys-profiler
        --profile-step-start "${PROFILE_STEP_START:-5}"
        --profile-step-end "${PROFILE_STEP_END:-7}"
        --profile-ranks 0
    )
fi

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype float32
)

MODEL_OVERRIDE_ARGS=(
    model.use_compile=$USE_COMPILE
    model.gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
    data.robot_config_root="$PROJECT_ROOT/../lingbot-vla-v2/configs/robot_configs"
)

# -- Logging params -----------------------------------------------------------
LOGGING_ARGS=(
    --log-interval 1
)

# -- Launch -------------------------------------------------------------------
echo "========================================================================"
echo "  LoongForge lingbot_vla_v2 Training (ZeRO-1)"
echo "  Model:        $MODEL_NAME"
echo "  GPUs:         $GPUS_PER_NODE x $NNODES nodes"
echo "  Micro batch:  $MICRO_BS"
echo "  Grad accum:   $GRAD_ACCUM  (global batch $(( MICRO_BS * GRAD_ACCUM * GPUS_PER_NODE * NNODES )))"
echo "  Data:         $DATA_PATH"
echo "  Output:       $OUTPUT_DIR"
echo "  Comm:         precision and overlap come from the model YAML"
echo "========================================================================"

cd "$AIAK_PATH"
PYTHONPATH=$AIAK_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    loongforge/embodied/train.py \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${MODEL_OVERRIDE_ARGS[@]}" \
    "$@"
