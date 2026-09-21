#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.6 SimplerEnv WidowX (Bridge) smoke.
# Default config: configs/simplerenv/widowx_stack_cube_smoke.yaml

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/groot_n1_6/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/simplerenv/widowx_stack_cube_smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIR:-/path/to/nvidia_lib}:/usr/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_lib/10_nvidia.json}
# Route the GR00T Eagle backbone through the repo-local Eagle3 builder (offline).
export CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-local}
export CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-full_iteration}

${BENCHMARK_PYTHON:-/path/to/conda/envs/simplerenv/bin/python} -m loongforge.evaluation.embodied.orchestrator.run --config "${CONFIG}"
