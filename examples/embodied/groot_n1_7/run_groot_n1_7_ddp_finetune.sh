#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_groot_n1_7_ddp_finetune.sh - GR00T-N1.7 Training Launch Script
#
# Usage:
#   bash run_groot_n1_7_ddp_finetune.sh
#   GPUS_PER_NODE=8 bash run_groot_n1_7_ddp_finetune.sh
#   bash run_groot_n1_7_ddp_finetune.sh --train-iters 500
#   bash run_groot_n1_7_ddp_finetune.sh model.action_horizon=32
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge/}"
export COSMOS_LOCAL_PATH="${COSMOS_LOCAL_PATH:-/workspace/huggingface.co/nvidia/Cosmos-Reason2-2B/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$COSMOS_LOCAL_PATH}"

# ── Paths ─────────────────────────────────────────────────────
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/huggingface.co/GR00T-N1.7-3B}"
DATA_PATH="${DATA_PATH:-/workspace/cube_to_bowl_5}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/}"
TENSORBOARD_PATH="${TENSORBOARD_PATH:-}"

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Model config ──────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name groot_n1_7
)

# ── Data config ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy groot_n1_7
    --dataset-path "$DATA_PATH"
    --lerobotdataset-version v2.1
    --video-backend torchcodec
    --num-workers 4
    --dataloader-multiprocessing-context fork
    --distributed-sampler-mode block
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type GrootN1d7Trainer
    --train-iters 100
    --per-device-batch-size 4
    --gradient-accumulation-steps 1
    --seed 42
    --output-dir "$OUTPUT_DIR"
    --lr-base 1.0e-4
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 5
    --min-lr 0.0
    #--optimizer AdamW
    --optimizer TEFusedAdamW
    --clip-grad 1.0
    --weight-decay 1.0e-5
    --weight-decay-grouping bias_norm
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --save-interval 200
    --pretrained-checkpoint "$CHECKPOINT_PATH"
    #--deterministic-mode
    --cuda-graph-impl local
    --cuda-graph-scope per_microbatch
    --cuda-graph-warmup-steps 3
    --cuda-graph-pad-length 0
    --no-cuda-graph-ddp-sync-in-graph
    --cuda-graph-grad-sync-bucket-mb 400
    --cuda-graph-grad-sync-impl coalesced
    --cuda-graph-grad-sync-dtype bf16
    --no-check-for-nan-in-loss-and-grad
)

# ── Distributed training ──────────────────────────────────────
DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --ddp-bucket-cap-mb 100
    --no-ddp-find-unused-parameters
    --ddp-static-graph
    # --dataloader-seed-workers
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --loss-log-rank -1
    --wandb-mode disabled
    --tensorboard-dir "$TENSORBOARD_PATH"
)

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge GR00T-N1.7 Training"
echo "  Model:      groot_n1_7"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────
PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
