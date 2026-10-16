#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Fill paths and evaluation settings in configs/libero/libero.yaml first.
# Default: four suites, all tasks, 50 episodes/task; edit the YAML to select a subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG=${CONFIG:-$SCRIPT_DIR/configs/libero/libero.yaml}
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$REPO_ROOT/$CONFIG"
fi

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-$REPO_ROOT/.venv-libero/bin/python}
export LIBERO_ROOT=${LIBERO_ROOT:-$REPO_ROOT/third_party/LIBERO}
export PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

exec "$BENCHMARK_PYTHON" -u -m loongforge.embodied.eval.orchestrator.run --config "$CONFIG"
