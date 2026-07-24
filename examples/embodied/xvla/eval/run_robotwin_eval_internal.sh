#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Internal one-click: X-VLA RoboTwin-2.0 task-success.
# Default (only shipped config): adjust_bottle_smoke_internal.yaml
# (X-VLA-RoboTwin2 + ee6d_dual). Optional knobs: comments in that YAML.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/xvla/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/adjust_bottle_smoke_internal.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/ssd1/opt/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/ssd1/opt/nvidia_lib/10_nvidia.json}

${BENCHMARK_PYTHON:-/workspace/miniconda3/envs/robotwin/bin/python} -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
