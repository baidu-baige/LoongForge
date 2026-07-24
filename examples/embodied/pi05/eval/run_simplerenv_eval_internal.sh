#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Internal one-click: SimplerEnv link smoke (random_init; not task-success).
# Task-success: xvla run_simplerenv_eval_internal.sh

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/simplerenv/widowx_stack_cube_smoke_internal.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/ssd1/opt/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/ssd1/opt/nvidia_lib/10_nvidia.json}

${BENCHMARK_PYTHON:-/workspace/miniconda3/envs/simplerenv/bin/python} -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
