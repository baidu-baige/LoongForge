#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_giga_brain_ddp_finetune.sh - GigaBrain-0 Training Launch Script (DDP)
#
# Reproduces the reference `giga-brain-0` run:
#   bash scripts/run_train.sh agibot_a2d_v30
# (configs/giga_brain_0_agibot_a2d_finetune_v30.py, which extends
#  configs/giga_brain_0_agibot_a2d_finetune.py) inside the LoongForge
# training framework (MODEL_REGISTRY + FinetuneTrainer), instead of
# giga_train.launch_from_config.
#
# Usage:
#   bash run_giga_brain_ddp_finetune.sh
#   TRAIN_ITERS=50 bash run_giga_brain_ddp_finetune.sh             # override via env
#   bash run_giga_brain_ddp_finetune.sh --lr-base 1e-4             # override a training param (flag form)
#   bash run_giga_brain_ddp_finetune.sh data.data_path=/other/path # override YAML fields (dotlist form)
#
# NOTE ON DISTRIBUTED STRATEGY: the reference run uses FSDP2 (3.5B params
# don't fit comfortably on one GPU alongside AdamW state). Set
# DISTRIBUTED_STRATEGY=ddp only if you have enough per-GPU memory headroom
# (e.g. a small-scale smoke test on 1-2 GPUs).
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29236"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Paths ─────────────────────────────────────────────────────
# Reference GB0_PRETRAINED / GB0_DATA_PATH_V30 / GB0_NORM_STATS_V30 (see
# giga-brain-0/scripts/run_train.sh + configs/giga_brain_0_agibot_a2d_finetune_v30.py).
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/ssd2/pxy/ckpt/open-gigaai/GigaBrain-0.1-3.5B-Base"}
DATA_PATH=${DATA_PATH:-"/ssd2/pxy/data/agibot-world/AgiBotWorldLerobot_v30_direct/agibotworld/task_327"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/giga_brain_ddp"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"giga_brain"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
# Reference: batch_size_per_gpu=32, num_workers=16 (dataloaders.train in
# giga_brain_0_agibot_a2d_finetune.py).
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format giga_brain_datasets
    --dataset-path "$DATA_PATH"
    --num-workers "$NUM_WORKERS"
)

# ── Training params ───────────────────────────────────────────
# Reference train.max_steps=1000 (finetune_v30 inherits finetune.py's default).
TRAIN_ITERS=${TRAIN_ITERS:-1000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SEED=${SEED:-6666}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate — reference optimizers.lr=2.5e-5 + schedulers=WarmupCosineScheduler
    # (warmup_steps=1000, decay_steps=30000, end_value=0.1). LoongForge's
    # cosine_with_min_lr decays base_lr -> min_lr (absolute), so
    # min_lr = lr_base * end_value = 2.5e-5 * 0.1 = 2.5e-6.
    --lr-base 2.5e-5
    --lr-decay-style cosine_with_min_lr
    --min-lr 2.5e-6
    --lr-warmup-iters 1000
    --lr-decay-iters 30000
    # Optimizer — reference optimizers=dict(betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 1e-10
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # EMA — reference train.with_ema=true (see optimizer/ema.py)
    --with-ema
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --pretrained-checkpoint "$CHECKPOINT_PATH"
)

# ── Distributed / precision ────────────────────────────────────
# Reference launch.distributed_type=FSDP, fsdp_config.fsdp_transformer_layer_cls_to_wrap=
# "SiglipEncoderLayer,Gemma2DecoderLayerWithExpert", train.mixed_precision='no'
# (giga_train applies bf16 internally via explicit autocast in the vendored
# PaliGemma2WithExpertModel/SiglipVisionTransformer forward — see
# model/giga_brain/paligemma2_with_expert.py — so LoongForge's outer --dtype
# only needs to cover the projection/loss layers that stay fp32 either way;
# bfloat16 matches the reference's effective compute dtype).
DISTRIBUTED_STRATEGY=${DISTRIBUTED_STRATEGY:-"fsdp"}
DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy "$DISTRIBUTED_STRATEGY"
    --dtype bfloat16
)
if [ "$DISTRIBUTED_STRATEGY" = "fsdp" ]; then
    DISTRIBUTED_TRAINING_ARGS+=(
        --fsdp-wrap-modules "SiglipEncoderLayer,Gemma2DecoderLayerWithExpert"
    )
fi

# Activation checkpointing — reference train.activation_checkpointing=true,
# activation_class_names=["SiglipEncoderLayer", "Gemma2DecoderLayerWithExpert"].
# LoongForge's --activation-checkpoint-module-patterns matches qualified
# module-key globs (dot-segments), not bare class names — both layer types
# live under model.paligemma_with_expert.{vision_tower.encoder,layers}.*.
ACTIVATION_CHECKPOINT_ARGS=(
    --activation-checkpoint-module-patterns \
        "model.paligemma_with_expert.vision_tower.encoder.layers.*,model.paligemma_with_expert.layers.*"
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge GigaBrain-0 Training (DDP/FSDP)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Strategy:   $DISTRIBUTED_STRATEGY"
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
    "${ACTIVATION_CHECKPOINT_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
