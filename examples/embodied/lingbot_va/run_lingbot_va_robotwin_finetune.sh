#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# LingBot-VA RobotWin post-training with baseline-aligned data semantics.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_NAME=lingbot_va_robotwin

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace/models/lingbot-va-posttrain-robotwin}
DATA_PATH=${DATA_PATH:-/workspace/datasets/robotwin-clean-and-aug-lerobot}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/outputs/lingbot_va_robotwin}

GRADIENT_ACCUMULATION_STEPS=1

source "$SCRIPT_DIR/lingbot_va_finetune_common.sh"
