#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run_pi05_ddp.sh - π₀.₅ VLA Training Launch Script (DDP, Single Node)
#
# Usage:
#   bash run_pi05_ddp.sh                                        # paligemma default
#   bash run_pi05_ddp.sh model.state_dim=8 data.image_size=448  # override YAML model:/data: fields (dotlist form)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="/workspace/LoongForge"
export NO_ALBUMENTATIONS_UPDATE=1

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/pi05_base"}
DATA_PATH=${DATA_PATH:-"/workspace/libero"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/workspace/tensorboard-log"}

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

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters 20
    --per-device-batch-size 12
    --gradient-accumulation-steps 1
    --seed 42
    --output-dir $OUTPUT_DIR
    # Learning rate
    --lr-base 2.5e-5
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 10
    --min-lr 1.0e-6
    # Optimizer
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval 10000
    --pretrained-checkpoint $CHECKPOINT_PATH
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --ddp-find-unused-parameters
    --ddp-static-graph
    --ddp-gradient-as-bucket-view
    --no-dynamo-optimize-ddp
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
