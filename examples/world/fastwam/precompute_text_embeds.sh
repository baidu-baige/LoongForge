#!/usr/bin/env bash
# Precompute FastWAM Text Embeddings
#
# This script pre-encodes text instructions for FastWAM (Fast World Action Model)
# datasets (e.g. LIBERO) into cached embedding files, so the text encoder does
# not run at every training step.
#
# Usage:
#   bash precompute_text_embeds.sh
#
# LOONGFORGE_PATH and LOCAL_VLA_ARTIFACTS_ROOT are set below and may be
# overridden from the environment; every variable below can also be
# overridden from the environment.
#
# Key environment variables (all optional, defaults shown resolve against
# $LOCAL_VLA_ARTIFACTS_ROOT):
#   DATASET_PATH             Root directory of the source dataset
#                            (default: $LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot)
#   TEXT_EMBEDDING_CACHE_DIR Where to write the embedding cache files
#                            (default: $LOCAL_VLA_ARTIFACTS_ROOT/fastwam/text_embeds)
#   MODEL_ID                 HuggingFace model ID for the text encoder
#                            (default: Wan-AI/Wan2.2-TI2V-5B)
#   TOKENIZER_MODEL_ID       HuggingFace model ID for the tokenizer
#                            (default: Wan-AI/Wan2.1-T2V-1.3B)
#   CONTEXT_LEN              Maximum token context length  (default: 128)
#   BATCH_SIZE               Encoding batch size           (default: 8)
#   DEVICE                   Compute device (cuda / cpu)   (default: cuda)
#   DTYPE                    Model dtype (bfloat16 / float16 / float32)
#                            (default: bfloat16)
#
# Example (override dataset path and cache dir):
#   DATASET_PATH=/data/libero \
#   TEXT_EMBEDDING_CACHE_DIR=/data/cache/fastwam_embeds \
#   bash precompute_text_embeds.sh
#
# Any extra arguments are forwarded directly to the Python script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_native_ci/vla_artifacts"}

# Point DiffSynth/Wan model loader at the shared local weights directory so
# text encoder + tokenizer are resolved from disk instead of Hugging Face Hub.
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/"}

DATASET_PATH=${DATASET_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot"}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/text_embeds"}
MODEL_ID=${MODEL_ID:-"Wan-AI/Wan2.2-TI2V-5B"}
TOKENIZER_MODEL_ID=${TOKENIZER_MODEL_ID:-"Wan-AI/Wan2.1-T2V-1.3B"}
CONTEXT_LEN=${CONTEXT_LEN:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}

mkdir -p "$TEXT_EMBEDDING_CACHE_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  FastWAM Precompute Text Embeddings"
echo "  Dataset:   $DATASET_PATH"
echo "  Cache:     $TEXT_EMBEDDING_CACHE_DIR"
echo "  Model:     $MODEL_ID"
echo "  Tokenizer: $TOKENIZER_MODEL_ID"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/loongforge/datasets/world/fastwam/transforms/precompute_text_embeds.py" \
    --dataset-root "$DATASET_PATH" \
    --output-dir "$TEXT_EMBEDDING_CACHE_DIR" \
    --model-id "$MODEL_ID" \
    --tokenizer-model-id "$TOKENIZER_MODEL_ID" \
    --context-len "$CONTEXT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "$@"

echo "FastWAM text embedding cache: $TEXT_EMBEDDING_CACHE_DIR"
