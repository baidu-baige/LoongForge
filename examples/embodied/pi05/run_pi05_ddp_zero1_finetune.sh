#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# sft_pi05_ddp.sh - π₀.₅ VLA SFT Launch Script (DDP, Embodied)
#
# Mirrors the config from examples/pi05/finetuning/sft_pi05.sh
# but uses the embodied training framework (loongforge/embodied/train.py).
#
# Usage:
#   bash sft_pi05_ddp.sh
#   bash sft_pi05_ddp.sh --train-iters 50000                   # override a training param (flag form)
#   bash sft_pi05_ddp.sh model.state_dim=8 data.image_size=448 # override YAML model:/data: fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="/workspace/LoongForge"

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/pi05_base"}
DATA_PATH=${DATA_PATH:-"/workspace/libero"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/workspace/tensorboard-log"}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
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
MODEL_NAME=${MODEL_NAME:-"pi05"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-path $DATA_PATH
    --tokenizer-path $TOKENIZER_PATH
    --robot-type libero_franka
    --num-workers 16
)

# ── Training params (aligned with examples/pi05/finetuning/sft_pi05.sh) ──
# Old Megatron config: micro-batch-size=16, global-batch-size=128, 8 GPUs
# => gradient-accumulation-steps = 128 / (16 * 8) = 1
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters 30000
    --per-device-batch-size 16
    --gradient-accumulation-steps 1
    --seed 1234
    --output-dir $OUTPUT_DIR
    # Learning rate
    --lr-base 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    # Optimizer
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval 30000
    --pretrained-checkpoint $CHECKPOINT_PATH
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --zero-optimizer
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge π₀.₅ Training (DDP)"
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
