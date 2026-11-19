#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_vpp2_sft_deepspeed_zero2_finetune.sh - VPP2 Action SFT Launch Script (ZeRO-2)
#
# Usage:
#   bash run_vpp2_sft_deepspeed_zero2_finetune.sh
#   TRAIN_ITERS=50 bash run_vpp2_sft_deepspeed_zero2_finetune.sh
#   bash run_vpp2_sft_deepspeed_zero2_finetune.sh --train-iters 500
#   bash run_vpp2_sft_deepspeed_zero2_finetune.sh model.action_horizon=16
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
PYTHON_BIN=${LOONGFORGE_PYTHON_BIN:-/usr/bin/python3}
export PYTHONNOUSERSITE=1
export SEED=${SEED:-42}
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Spawned data workers should not wait for online package version checks.
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export VPP2_DETERMINISTIC=${VPP2_DETERMINISTIC:-1}

# Main-chain performance switches, migrated to typed LoongForge configuration
# below. These defaults reproduce the validated 2.6360 samples/s/gpu recipe.
export VPP2_VIDEO_PREFILL_UNMASKED_SDPA=${VPP2_VIDEO_PREFILL_UNMASKED_SDPA:-1}
export VPP2_ACTION_VISIBLE_KV_SDPA=${VPP2_ACTION_VISIBLE_KV_SDPA:-1}

export VPP2_ASSETS_ROOT=/workspace/vpp2/vpp2_assets

# ── Distributed ───────────────────────────────────────────────
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29531"}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Paths ─────────────────────────────────────────────────────
export VPP2_VIDEO_CHECKPOINT=${VPP2_VIDEO_CHECKPOINT:-"$VPP2_ASSETS_ROOT/libero_video_step010000.pt"}
export VPP2_DATASET_ROOT=${VPP2_DATASET_ROOT:-"$VPP2_ASSETS_ROOT/libero_subset"}
DATASET_PATH=${DATASET_PATH:-"$VPP2_DATASET_ROOT/libero_spatial_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_object_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_goal_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_10_no_noops_lerobot"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vpp2_deepspeed_zero2/$(date +%Y%m%d_%H%M%S_%N)"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$LOONGFORGE_PATH/examples/embodied/vpp2/deepspeed_libero_zero2.json"}

# ── Training defaults ─────────────────────────────────────────
export TRAIN_ITERS=${TRAIN_ITERS:-50}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
# An explicit horizon can outlive an early stop; otherwise the scheduler uses
# the final --train-iters value, including a trailing CLI override.
SCHEDULE_ARGS=()
if [[ -n "${SCHEDULE_STEPS:-}" ]]; then
    SCHEDULE_ARGS+=(--lr-decay-iters "$SCHEDULE_STEPS")
fi
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
# Prepare the next batch while the GPU trains. With zero workers, LIBERO
# decoding and preprocessing run synchronously in the training process.
export NUM_WORKERS=1

# Highest-throughput LoongForge configuration measured in the local 50-step
# logs: native frozen CLIP/VAE graphs plus Action Graph visible-K/V compaction.
export VPP2_COMPILE_CLIP=${VPP2_COMPILE_CLIP:-native_graph}
export VPP2_COMPILE_VAE_ENCODE=${VPP2_COMPILE_VAE_ENCODE:-native_graph}
export VPP2_ACTION_GRAPH_VISIBLE_KV=${VPP2_ACTION_GRAPH_VISIBLE_KV:-true}
VPP2_ACTION_EXECUTION=${VPP2_ACTION_EXECUTION:-native_graph}

# The all-True mask-elision candidate is tied to the captured native graph.
# Eager mode still uses the migrated unmasked-SDPA switch, but does not enable
# this separate graph-specific candidate by accident.
if [[ -z "${VPP2_VIDEO_ATTENTION_MASK_ELISION+x}" ]]; then
    if [[ "$VPP2_ACTION_EXECUTION" == native_graph ]]; then
        VPP2_VIDEO_ATTENTION_MASK_ELISION=true
    else
        VPP2_VIDEO_ATTENTION_MASK_ELISION=false
    fi
fi
# Capture before the step1 timer without consuming data or updating weights.
export VPP2_WARMUP_BEFORE_TRAINING=${VPP2_WARMUP_BEFORE_TRAINING:-true}

# ── Runtime options ────────────────────────────────────────────
export PYTHONPATH="$LOONGFORGE_PATH${PYTHONPATH:+:$PYTHONPATH}"
DETERMINISTIC_ARGS=()
if [[ "$VPP2_DETERMINISTIC" == 1 ]]; then
    DETERMINISTIC_ARGS+=(--deterministic-mode)
fi

CONFIG_OVERRIDES=(
    model.disable_train_autocast=true
    "model.initialization_seed=$SEED"
    # Accepted frozen-video/ActionDiT performance recipe.
    model.perf_video_execution=eager
    "model.perf_action_execution=$VPP2_ACTION_EXECUTION"
    model.perf_video_rope_fused=true
    "model.perf_video_prefill_unmasked_sdpa=$VPP2_VIDEO_PREFILL_UNMASKED_SDPA"
    model.perf_video_native_layer_norm=true
    model.perf_video_rms_pointwise_fused=true
    "model.perf_video_attention_mask_elision=$VPP2_VIDEO_ATTENTION_MASK_ELISION"
    model.perf_video_skip_uninitialized_fill=true
    "model.perf_compile_clip='$VPP2_COMPILE_CLIP'"
    "model.perf_compile_vae_encode='$VPP2_COMPILE_VAE_ENCODE'"
    "model.perf_action_visible_kv_sdpa=$VPP2_ACTION_VISIBLE_KV_SDPA"
    "model.perf_action_graph_visible_kv=$VPP2_ACTION_GRAPH_VISIBLE_KV"
    "model.perf_warmup_before_training=$VPP2_WARMUP_BEFORE_TRAINING"
)

# ── Model config ──────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name vpp2
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy vpp2
    --dataset-path "$DATASET_PATH"
    --video-backend pyav
    --num-workers "$NUM_WORKERS"
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --set-seed-by-rank
    --output-dir "$OUTPUT_DIR"
    # Learning rate: stopping early must not shorten the reference schedule.
    --lr-base 1.0e-4
    --lr-decay-style linear_warmup_cosine_annealing
    --lr-warmup-iters 2
    "${SCHEDULE_ARGS[@]}"
    --min-lr 1.0e-6
    # Optimizer: preserve the complete logical batch and RNG draw order.
    --optimizer AdamW
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 0.0
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
    --save-format pt
    --no-check-for-nan-in-loss-and-grad
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy deepspeed
    --deepspeed-config "$DEEPSPEED_CONFIG"
    --dtype bfloat16
    --no-init-on-meta
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge VPP2 Action Training (DeepSpeed ZeRO-2)"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Model:      vpp2"
echo "  Checkpoint: $VPP2_VIDEO_CHECKPOINT"
echo "  Data:       $DATASET_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

launch_training() {
    # argparse's trailing dotlist positional must follow all option arguments.
    local argument
    local -a user_training_args=() user_config_args=()
    for argument in "$@"; do
        case "$argument" in
            model.*=*|data.*=*) user_config_args+=("$argument") ;;
            *) user_training_args+=("$argument") ;;
        esac
    done
    exec "$PYTHON_BIN" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
        "${MODEL_CONFIG_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DISTRIBUTED_TRAINING_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        "${DETERMINISTIC_ARGS[@]}" \
        "${user_training_args[@]}" \
        "${CONFIG_OVERRIDES[@]}" \
        "${user_config_args[@]}"
}

launch_training "$@"
