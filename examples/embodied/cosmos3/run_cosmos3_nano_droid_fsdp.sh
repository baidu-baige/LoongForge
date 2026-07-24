#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_cosmos3_nano_droid_fsdp.sh - Cosmos3-Nano DROID Action-Policy SFT (FSDP2)
#
# DROID joint_pos 8D + use_state action policy SFT, mirroring cosmos's
# launch_sft_action_policy_droid.sh recipe.
#
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="${LOONGFORGE_PATH:-$(realpath "$SCRIPT_DIR/../../..")}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen/Qwen3-VL-8B-Instruct"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/checkpoints/Cosmos3-Nano"}
DATASET_PATH=${DATASET_PATH:-"/mnt/cluster/datasets/nvidia/Cosmos3-DROID/success/"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/cosmos3_nano_droid_fsdp_$(date +%Y%m%d_%H%M%S)"}

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29510"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"cosmos3_nano"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy cosmos3_droid
    --dataset-path $DATASET_PATH
    --tokenizer-path $TOKENIZER_PATH
    --num-workers 4
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --init-on-meta
    --trainer-type FinetuneTrainer
    --train-iters 500
    --per-device-batch-size 2
    --gradient-accumulation-steps 1
    --seed 42
    #--deterministic-mode
    --disable-tf32
    --output-dir $OUTPUT_DIR
    --pretrained-checkpoint $CHECKPOINT_PATH
    --lr-group net.action2llm=1e-3,net.llm2action=1e-3,net.action_modality_embed=1e-3,net=2e-4
    --lr-decay-style lambda_linear
    --lr-warmup-iters 0
    --optimizer TorchFusedAdamW
    --clip-grad 1.0
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-8
    --save-interval 100
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy fsdp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-project cosmos3-nano
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Cosmos3-Nano DROID Action-Policy SFT (FSDP2)"
echo "  Model:      $MODEL_NAME"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Data:       $DATASET_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

export PYTHONHASHSEED=42

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
