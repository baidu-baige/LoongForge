#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
GLM_HF=${GLM_HF:-/mnt/cluster/huggingface.co/GLM/GLM-5.2}
KIMI_HF=${KIMI_HF:-/mnt/cluster/huggingface.co/moonshotai/Kimi-K2.6}
OUTPUT=${OUTPUT:-/mnt/cluster/LoongForge/GLM/GLM-5.2-Kimi-K2.6-ViT-base-hf}

python3 \
  "$LOONGFORGE_PATH/examples/glm5.2_vit/checkpoint_convert/compose_glm5_2_kimi_k2_6_hf.py" \
  --glm-hf "$GLM_HF" \
  --kimi-hf "$KIMI_HF" \
  --output "$OUTPUT"
