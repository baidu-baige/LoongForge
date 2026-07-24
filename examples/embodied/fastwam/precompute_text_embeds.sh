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
# Key environment variables (all optional, defaults shown):
#   LOONGFORGE_PATH          Path to the LoongForge repo root
#                            (default: /workspace/LoongForge)
#   DATASET_PATH             Root directory of the source dataset
#                            (default: /path/to/libero)
#   TEXT_EMBEDDING_CACHE_DIR Where to write the embedding cache files
#                            (default: $LOONGFORGE_PATH/data/fastwam_text_embeds)
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

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

DATASET_PATH="${DATASET_PATH:-/path/to/libero}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-$LOONGFORGE_PATH/data/fastwam_text_embeds}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/loongforge/embodied/data/datasets/fastwam/transforms/precompute_text_embeds.py" \
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
