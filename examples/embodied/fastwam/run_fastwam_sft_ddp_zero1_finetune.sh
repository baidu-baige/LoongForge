#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_fastwam_sft_ddp_zero1_finetune.sh - FastWAM SFT Launch Script (DDP + ZeRO-1)
#
# Everything is driven by environment variables; the script forwards "$@" to
# train.py. Config knobs that are not CLI flags must go through the dotlist
# escape hatches below (EXTRA_MODEL_OVERRIDES / EXTRA_DATA_OVERRIDES) — argparse
# rejects positional overrides placed after the optionals this script emits.
#
# ── Official-aligned recipe (matches standalone FastWAM) ──────────────────────
# LR/clip/sampler/optimizer alignment:
#   LR_DECAY_STYLE=linear_warmup_cosine_annealing   (official 5% warmup + cosine, eta_min=lr*0.01)
#   SAMPLER_EPOCH_MODE=fixed                        (official ResumableEpochSampler never advances its epoch)
#   ZERO_MASTER_PARAM_DTYPE=fp32                    (official DeepSpeed BF16_Optimizer keeps fp32 masters)
#   CLIP_GRAD=0                                     (official effectively does not clip; see PR #274)
# Config knobs (only needed when not already in your --config-file):
#   EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none"
#   EXTRA_DATA_OVERRIDES="data.norm_stats_path=/path/to/dataset_stats.json data.normalization_mode=min_max"
# e.g. the recipe validated for LIBERO (4-suite joint, 20k, fp32 master, accel on):
#   LR_DECAY_STYLE=linear_warmup_cosine_annealing CLIP_GRAD=0 SAMPLER_EPOCH_MODE=fixed \
#   ZERO_MASTER_PARAM_DTYPE=fp32 SAVE_FORMAT=dcp ASYNC_SAVE=1 \
#   EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none" \
#   EXTRA_DATA_OVERRIDES="data.norm_stats_path=/path/to/dataset_stats.json" \
#   bash run_fastwam_sft_ddp_zero1_finetune.sh --lr-base 1e-4 --lr-warmup-iters 1000 --min-lr 1e-6
# The "faithful" variant (acceleration switches off) additionally sets
#   EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none model.disable_train_autocast=false model.drop_all_true_cross_attn_mask=false"
#   EXTRA_DATA_OVERRIDES="data.image_normalize_mode=identity data.image_resize_strategy=resize_only data.normalization_mode=min_max"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/"}

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29519"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/Wan2.2-TI2V-5B"}
DATASET_PATH=${DATASET_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/fastwam_sft_zero1"}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/text_embeds"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"fastwam"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy fastwam
    --dataset-path "$DATASET_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type libero_franka
    --num-workers "$NUM_WORKERS"
    --lerobotdataset-version v2.1
    --video-backend pyav
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
SEED=${SEED:-3047}

# ── Alignment knobs ───────────────────────────────────────────
# Defaults reproduce the previous behaviour exactly; the official-aligned recipe
# (see the header) sets them via the environment. The three opt-in flags below are
# only appended to the command line when set, so an unset invocation produces the
# byte-identical argv it always did.
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine_warmup_with_min_lr}
OPTIMIZER=${OPTIMIZER:-TorchFusedAdamW}
CLIP_GRAD=${CLIP_GRAD:-1.0}
SAMPLER_EPOCH_MODE=${SAMPLER_EPOCH_MODE:-}
ZERO_MASTER_PARAM_DTYPE=${ZERO_MASTER_PARAM_DTYPE:-}
SAVE_FORMAT=${SAVE_FORMAT:-}
ASYNC_SAVE=${ASYNC_SAVE:-0}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1.0e-8
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-iters 0
    --min-lr 1.0e-9
    # Optimizer
    --optimizer "$OPTIMIZER"
    --cudnn-benchmark
    --clip-grad "$CLIP_GRAD"
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
)
# Opt-in: appended only when set, so the default argv is unchanged.
if [[ -n "$SAVE_FORMAT" ]]; then TRAINING_ARGS+=(--save-format "$SAVE_FORMAT"); fi
if [[ -n "$SAMPLER_EPOCH_MODE" ]]; then TRAINING_ARGS+=(--sampler-epoch-mode "$SAMPLER_EPOCH_MODE"); fi
if [[ -n "$ZERO_MASTER_PARAM_DTYPE" ]]; then TRAINING_ARGS+=(--zero-master-param-dtype "$ZERO_MASTER_PARAM_DTYPE"); fi
if [[ "$ASYNC_SAVE" == "1" ]]; then TRAINING_ARGS+=(--async-save); fi

if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
    TRAINING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --zero-optimizer
    --no-ddp-find-unused-parameters
    --ddp-static-graph
    --ddp-gradient-as-bucket-view
    --no-ddp-broadcast-buffers
    --ddp-bucket-cap-mb 200
    --dtype bfloat16
    --no-check-for-nan-in-loss-and-grad
    --zero-parameters-as-bucket-view
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Model/data dotlist overrides ──────────────────────────────
MODEL_DATA_OVERRIDES=()
if [[ -n "$ACTION_DIT_PRETRAINED_PATH" ]]; then
    MODEL_DATA_OVERRIDES+=("model.action_dit_pretrained_path=$ACTION_DIT_PRETRAINED_PATH")
fi
if [[ -n "$TEXT_EMBEDDING_CACHE_DIR" ]]; then
    MODEL_DATA_OVERRIDES+=("data.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE_DIR")
fi

# Generic config escape hatches: space-separated dotlist overrides, e.g.
#   EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none"
#   EXTRA_DATA_OVERRIDES="data.norm_stats_path=/path/to/dataset_stats.json data.normalization_mode=min_max"
# They must be injected here rather than passed on the command line: this script
# appends "$@" after its optionals, and argparse rejects positional `overrides`
# there ("unrecognized arguments").
if [[ -n "${EXTRA_MODEL_OVERRIDES:-}" ]]; then
    read -r -a _extra_model_overrides <<< "$EXTRA_MODEL_OVERRIDES"
    MODEL_DATA_OVERRIDES+=("${_extra_model_overrides[@]}")
fi
if [[ -n "${EXTRA_DATA_OVERRIDES:-}" ]]; then
    read -r -a _extra_data_overrides <<< "$EXTRA_DATA_OVERRIDES"
    MODEL_DATA_OVERRIDES+=("${_extra_data_overrides[@]}")
fi

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (DDP + ZeRO-1)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Data:       $DATASET_PATH"
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
    "${MODEL_DATA_OVERRIDES[@]+"${MODEL_DATA_OVERRIDES[@]}"}" \
    "$@"
