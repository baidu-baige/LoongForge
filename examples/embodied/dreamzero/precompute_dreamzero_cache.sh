#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Precompute DreamZero frozen-condition features for structured cache artifacts.
#
# Usage:
#   bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh
#
#   MODEL_NAME=dreamzero_full_wan21_14b \
#     DATA_PATH=/path/to/droid_lerobot \
#     WAN21_CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P \
#     CACHE_OUTPUT_DIR=/path/to/dreamzero_cache \
#     GPUS_PER_NODE=8 \
#     bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh
#
#   NUM_SAMPLES=8 DRY_RUN=1 \
#     bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh
#
# Set VALIDATION_REQUIRE_FULL_COVERAGE=1 for production artifacts that must
# cover the entire dataset. Selected-index and small smoke caches validate the
# requested selection by default.
#
# Wan2.2-5B defaults to video latents + raw prompt embeddings. Wan2.1-14B
# defaults to video latents + first-frame latents. Additional precompute tool
# arguments can be appended directly, for example --no-include-prompt-embs.
# The tool prints the model.precomputed_cache.* overrides required by training.

set -euo pipefail

# ── Paths ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$PROJECT_ROOT"}

DREAMZERO_DATA_ROOT=${DREAMZERO_DATA_ROOT:-"/workspace/dreamzero/data"}
DREAMZERO_CKPT_ROOT=${DREAMZERO_CKPT_ROOT:-"/workspace/dreamzero/checkpoints"}
DREAMZERO_CACHE_ROOT=${DREAMZERO_CACHE_ROOT:-"/workspace/dreamzero/cache"}

# ── Model and data ────────────────────────────
MODEL_NAME=${MODEL_NAME:-dreamzero_full_wan22_5b}
case "$MODEL_NAME" in
    dreamzero_full_wan22_5b|dreamzero_lora_wan22_5b)
        DEFAULT_CONFIG_NAME=dreamzero_wan22_5b
        ;;
    dreamzero_full_wan21_14b|dreamzero_lora_wan21_14b)
        DEFAULT_CONFIG_NAME=dreamzero_wan21_14b
        ;;
    *)
        DEFAULT_CONFIG_NAME=$MODEL_NAME
        ;;
esac
CONFIG_FILE=${CONFIG_FILE:-"$PROJECT_ROOT/configs/models/embodied/${DEFAULT_CONFIG_NAME}.yaml"}

export WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_CKPT_ROOT/Wan2.1-I2V-14B-480P"}
export WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_CKPT_ROOT/Wan2.2-TI2V-5B"}
export DREAMZERO_AGIBOT_CKPT_DIR=${DREAMZERO_AGIBOT_CKPT_DIR:-"$DREAMZERO_CKPT_ROOT/DreamZero-AgiBot"}

case "$MODEL_NAME" in
    *libero*)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/libero_lerobot"
        ;;
    *agibot*)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/agibot_lerobot"
        ;;
    *yam*)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/yam_lerobot"
        ;;
    *)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/droid_lerobot"
        ;;
esac

case "$MODEL_NAME" in
    *wan22_5b*)
        TOKENIZER_PATH=${TOKENIZER_PATH:-"$WAN22_CKPT_DIR/google/umt5-xxl"}
        ;;
    *wan21_14b*)
        TOKENIZER_PATH=${TOKENIZER_PATH:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
        ;;
    *)
        echo "Unsupported DreamZero MODEL_NAME: $MODEL_NAME" >&2
        echo "Expected a model name containing wan21_14b or wan22_5b." >&2
        exit 2
        ;;
esac

DATA_PATH=${DATA_PATH:-"$DEFAULT_DATA_PATH"}
CACHE_OUTPUT_DIR=${CACHE_OUTPUT_DIR:-"$DREAMZERO_CACHE_ROOT/$MODEL_NAME"}

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "DreamZero config not found: $CONFIG_FILE" >&2
    exit 2
fi

# ── Distributed ──────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-${NUM_GPUS:-1}}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

# ── Cache generation ─────────────────────────
START_INDEX=${START_INDEX:-0}
NUM_SAMPLES=${NUM_SAMPLES:-1000000000}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
STORAGE_FORMAT=${STORAGE_FORMAT:-tensor_shards}
TENSOR_SHARD_SIZE=${TENSOR_SHARD_SIZE:-4096}
DTYPE=${DTYPE:-bf16}

# Fixed per-sample transform seeds make generated features reproducible.
USE_SAMPLE_TRANSFORM_SEED=${USE_SAMPLE_TRANSFORM_SEED:-1}
SAMPLE_TRANSFORM_SEED=${SAMPLE_TRANSFORM_SEED:-0}
# Exclude samples that cannot provide a complete language-conditioned chunk.
REQUIRE_FULL_LANGUAGE_CHUNKS=${REQUIRE_FULL_LANGUAGE_CHUNKS:-1}
VALIDATE_CACHE=${VALIDATE_CACHE:-1}
VALIDATION_SAMPLE_COUNT=${VALIDATION_SAMPLE_COUNT:-8}
VALIDATION_REQUIRE_FULL_COVERAGE=${VALIDATION_REQUIRE_FULL_COVERAGE:-0}

is_enabled() {
    local value=${1,,}
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

PRECOMPUTE_ARGS=(
    --config-file "$CONFIG_FILE"
    --data-path "$DATA_PATH"
    --output-dir "$CACHE_OUTPUT_DIR"
    --start-index "$START_INDEX"
    --num-samples "$NUM_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --storage-format "$STORAGE_FORMAT"
    --tensor-shard-size "$TENSOR_SHARD_SIZE"
    --tokenizer-path "$TOKENIZER_PATH"
    --dtype "$DTYPE"
)

if is_enabled "$USE_SAMPLE_TRANSFORM_SEED"; then
    PRECOMPUTE_ARGS+=(--use-sample-transform-seed --sample-transform-seed "$SAMPLE_TRANSFORM_SEED")
else
    PRECOMPUTE_ARGS+=(--no-use-sample-transform-seed)
fi

if is_enabled "$REQUIRE_FULL_LANGUAGE_CHUNKS"; then
    PRECOMPUTE_ARGS+=(--require-full-language-chunks)
else
    PRECOMPUTE_ARGS+=(--no-require-full-language-chunks)
fi

# ── Launch ──────────────────────────────────
CMD=(
    torchrun "${DISTRIBUTED_ARGS[@]}"
    "$LOONGFORGE_PATH/tools/data_preprocess/embodied/dreamzero/precompute_features.py"
    "${PRECOMPUTE_ARGS[@]}"
    "$@"
)

echo "========================================================================"
echo "  LoongForge DreamZero Cache Precompute"
echo "  Model:      $MODEL_NAME"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES nodes"
echo "  Data:       $DATA_PATH"
echo "  Cache:      $CACHE_OUTPUT_DIR"
echo "========================================================================"

if is_enabled "${DRY_RUN:-0}"; then
    printf 'PYTHONPATH=%q ' "$LOONGFORGE_PATH:${PYTHONPATH:-}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" "${CMD[@]}"

if is_enabled "$VALIDATE_CACHE" && [[ "$NODE_RANK" == "0" ]]; then
    VALIDATION_ARGS=(
        --manifest "$CACHE_OUTPUT_DIR/manifest.json"
        --cache-dir "$CACHE_OUTPUT_DIR"
        --expect-sample-transform-seed "$SAMPLE_TRANSFORM_SEED"
        --sample-count "$VALIDATION_SAMPLE_COUNT"
    )
    if is_enabled "$VALIDATION_REQUIRE_FULL_COVERAGE"; then
        VALIDATION_ARGS+=(--require-full-coverage)
    fi
    if is_enabled "$USE_SAMPLE_TRANSFORM_SEED"; then
        VALIDATION_ARGS+=(--expect-use-sample-transform-seed true)
    else
        VALIDATION_ARGS+=(--expect-use-sample-transform-seed false)
    fi

    PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
        python "$LOONGFORGE_PATH/tools/data_preprocess/embodied/dreamzero/validate_precomputed_feature_artifact.py" \
        "${VALIDATION_ARGS[@]}"
fi

if [[ "$NODE_RANK" == "0" ]]; then
    echo "DreamZero cache artifact: $CACHE_OUTPUT_DIR"
fi
