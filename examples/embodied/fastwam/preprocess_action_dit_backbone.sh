#!/usr/bin/env bash
# Preprocess ActionDiT Backbone Weights for FastWAM
#
# This script extracts and reshapes the video DiT backbone weights from a
# Wan2.2 checkpoint into the ActionDiT format required by FastWAM training.
# The result is a single .pt file that can be loaded as the action backbone
# during LoongForge FastWAM training runs.
#
# Run this once before starting FastWAM training whenever you change the
# base Wan2.2 model.
#
# Usage:
#   bash preprocess_action_dit_backbone.sh
#
# Key environment variables (all optional, defaults shown):
#   LOONGFORGE_PATH      Path to the LoongForge repo root
#                        (default: /workspace/LoongForge)
#   MODEL_ID             HuggingFace model ID for the Wan2.2 video DiT
#                        (default: Wan-AI/Wan2.2-TI2V-5B)
#   TOKENIZER_MODEL_ID   HuggingFace model ID for the tokenizer
#                        (default: Wan-AI/Wan2.1-T2V-1.3B)
#   OUTPUT               Path where the processed .pt backbone file is saved
#                        (default: $LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt)
#   DEVICE               Compute device (cpu / cuda)   (default: cpu)
#   DTYPE                Model dtype (float32 / bfloat16 / float16)
#                        (default: float32)
#   LOCAL_MODEL_PATH     Local root directory for model artifacts; files are
#                        looked up at <LOCAL_MODEL_PATH>/<model-id>/.
#                        Leave empty to download from HuggingFace Hub.
#                        Example: LOCAL_MODEL_PATH=/data/models
#
# Example (use a locally cached model):
#   LOCAL_MODEL_PATH=/data/models \
#   OUTPUT=/data/checkpoints/action_dit_backbone.pt \
#   bash preprocess_action_dit_backbone.sh
#
# Any extra arguments are forwarded directly to the Python script.

set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
OUTPUT="${OUTPUT:-$LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
# Local root directory for model artifacts; files are looked up at <LOCAL_MODEL_PATH>/<model-id>/.
# Example: LOCAL_MODEL_PATH=/data/models 
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/loongforge/embodied/data/datasets/fastwam/transforms/preprocess_action_dit_backbone.py" \
    --output "$OUTPUT" \
    --model-id "$MODEL_ID" \
    --tokenizer-model-id "$TOKENIZER_MODEL_ID" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    "$@"

echo "ActionDiT backbone payload saved to: $OUTPUT"
