#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run_xvla_ddp_finetune.sh - X-VLA Training Launch Script (DDP, Single Node)
#
# Usage:
#   bash run_xvla_ddp_finetune.sh
#   bash run_xvla_ddp_finetune.sh --lr 1e-4                              # override training param
#   bash run_xvla_ddp_finetune.sh backbone.image_size=448                # override YAML field
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="/workspace/AIAK-Training-Omni"

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/ckpt/X-VLA-WidowX"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/X-VLA-WidowX"}
export DATA_PATH=${DATA_PATH:-"/workspace/data/XVLA-Soft-Fold/0928_10am_new/"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/outputs/xvla_ddp"}
MASTER_PORT=29235
GRADIENT_ACCUMULATION_STEPS=1

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
echo "GPUS_PER_NODE="$GPUS_PER_NODE
echo "NNODES="$NNODES

# nsys profiling is OFF by default — wrapping torchrun with `nsys profile` makes
# the run appear to "hang" after the last training step while nsys collects and
# writes the .nsys-rep report (slow for multi-view VLA, and it also waits on
# re-parented child processes). Enable explicitly with XVLA_PROFILE=1.
if [ "${XVLA_PROFILE:-0}" = "1" ]; then
    NSYS_ARGS="nsys profile \
        --output=${OUTPUT_DIR}/nsys_report \
        -s none --trace=cuda,nvtx,osrt \
        --force-overwrite=true"
else
    NSYS_ARGS=""
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"xvla"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format hdf5_datasets
    --dataset-path $DATA_PATH
    --tokenizer-path $TOKENIZER_PATH
    --robot-type libero_franka
    --num-workers 2
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters 40
    --per-device-batch-size 16
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed 42
    --output-dir $OUTPUT_DIR
    --lr-base 0.0001
    --lr-group "model.vlm=1e-5,model.transformer.soft_prompt_hub=1e-5"
    --lr-warmup-iters 5
    --loss-spike-threshold 1000
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval 100
    --pretrained-checkpoint $CHECKPOINT_PATH
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 20
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge X-VLA Training (DDP)"
echo "  Model:      $MODEL_NAME"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    $NSYS_ARGS \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"   # pass-through: --lr 1e-4  OR  backbone.image_size=448
