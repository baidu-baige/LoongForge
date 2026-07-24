#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_groot_n1_6_ddp.sh - GR00T-N1.6 Training Launch Script
#
# Usage:
#   bash run_groot_n1_6_ddp.sh
#   GPUS_PER_NODE=8 bash run_groot_n1_6_ddp.sh
#   bash run_groot_n1_6_ddp.sh --train-iters 50000            # override a training param (flag form)
#   bash run_groot_n1_6_ddp.sh model.tune_llm=true            # override YAML model:/data: fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"
export EAGLE_LOCAL_PATH=${EAGLE_LOCAL_PATH:-"/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6"}

# ── Paths ─────────────────────────────────────────────────────
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/huggingface.co/nvidia/GR00T-N1.6-3B"}
DATA_PATH=${DATA_PATH:-"/workspace/libero_object_no_noops_1.0.0_lerobot_3.0"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/groot_n1_6"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/workspace/tensorboard-log/groot_n1_6"}


# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}


DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"groot_n1_6"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)


DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-path $DATA_PATH
    --robot-type libero_franka
    --video-backend torchcodec
    --num-workers 16
    --dataloader-multiprocessing-context spawn
    --distributed-sampler-mode block
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type GrootN1d6Trainer
    --train-iters 50
    --per-device-batch-size 16
    --gradient-accumulation-steps 1
    --seed 1234
    --output-dir $OUTPUT_DIR
    --lr-base 1.0e-4
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 2
    --min-lr 0.0
    --optimizer TEFusedAdamW
    --clip-grad 1.0
    --weight-decay 1.0e-5
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    # Checkpoint
    --save-interval 200
    --pretrained-checkpoint $CHECKPOINT_PATH
    # Determinism / validation checks
    --deterministic-mode
    --cuda-graph-impl local
    --cuda-graph-scope per_microbatch
    --cuda-graph-warmup-steps 3
    --cuda-graph-pad-length 220
    --no-cuda-graph-ddp-sync-in-graph
    --cuda-graph-grad-sync-bucket-mb 400
    --cuda-graph-grad-sync-impl coalesced
    --cuda-graph-grad-sync-dtype bf16
    --no-check-for-nan-in-loss-and-grad
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    #--ddp-find-unused-parameters
    #--ddp-gradient-as-bucket-view
    --ddp-static-graph
)
 
# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "$TENSORBOARD_PATH"
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge GR00TN1.6 Training (DDP)"
echo "  Model:      $MODEL_NAME"
echo "  GPUs:       $GPUS_PER_NODE"
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
    "$@"
