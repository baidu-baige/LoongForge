#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# LingBot-VA LIBERO-Long post-training with baseline-aligned data semantics.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME=lingbot_va_libero

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace/models/lingbot-va-posttrain-libero-long}
DATA_PATH=${DATA_PATH:-/workspace/datasets/libero-long-lerobot}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/outputs/lingbot_va_libero}

GRADIENT_ACCUMULATION_STEPS=10

source "$SCRIPT_DIR/lingbot_va_finetune_common.sh"
