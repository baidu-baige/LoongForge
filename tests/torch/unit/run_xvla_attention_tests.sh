#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Runs the XVLA attention forward unit tests.
# Self-contained: no dataset / checkpoint / external files required (a CUDA GPU is).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python -m pytest "$SCRIPT_DIR/test_xvla_attention_forward.py" -v "$@"
