#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_cosmos3_nano_droid_fsdp_finetune.sh - Cosmos3-Nano DROID Action-Policy SFT (FSDP2)
#
# DROID joint_pos 8D + use_state action policy SFT, mirroring cosmos's
# launch_sft_action_policy_droid.sh recipe.
#
# Usage:
#   bash run_cosmos3_nano_droid_fsdp_finetune.sh
#   TRAIN_ITERS=50 bash run_cosmos3_nano_droid_fsdp_finetune.sh    # override via env
#   bash run_cosmos3_nano_droid_fsdp_finetune.sh --train-iters 50  # override a training param (flag form)
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
MASTER_PORT=${MASTER_PORT:-"29510"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/cosmos3/tokenizers/Qwen3-VL-8B-Instruct"}
VAE_PATH=${VAE_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/cosmos3/models/Wan2.2_VAE/Wan2.2_VAE.pth"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/cosmos3/models/Cosmos3-Nano"}
DATA_PATH=${DATA_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/cosmos3/datasets/Cosmos3-DROID-subset"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/cosmos3_nano"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"cosmos3_nano"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-4}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy cosmos3_droid
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --num-workers "$NUM_WORKERS"
    #--no-sampler-shuffle
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-500}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
SEED=${SEED:-42}
export PYTHONHASHSEED=$SEED
export COSMOS3_TV_TRANSFORMS=v2

TRAINING_ARGS=(
    --init-on-meta
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --set-seed-by-rank
    #--deterministic-mode
    --disable-tf32
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-group net.action2llm=1e-3,net.llm2action=1e-3,net.action_modality_embed=1e-3,net=2e-4
    --lr-decay-style lambda_linear
    --lr-warmup-iters 0
    # Optimizer
    --optimizer TorchFusedAdamW
    --clip-grad 1.0
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-8
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --pretrained-checkpoint "$CHECKPOINT_PATH"
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy fsdp
    --dtype bfloat16
    --fsdp-reduce-dtype bf16
    --fsdp-wrap-modules MoTDecoderLayer
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
    "model.qwen3_vl_path=$TOKENIZER_PATH" \
    "model.vae_path=$VAE_PATH" \
    "$@"
