#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Runs the XVLA attention forward unit tests.
# Self-contained: no dataset / checkpoint / external files required (a CUDA GPU is).
set -euo pipefail

LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/AIAK-Training-Omni}"
TEST_FILE="$LOONGFORGE_PATH/loongforge/embodied/model/xvla/tests/test_attention_forward.py"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
    python -m pytest "$TEST_FILE" -v "$@"
