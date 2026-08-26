#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.7 SimplerEnv WidowX (Bridge) task-success.
# Default config: configs/simplerenv/widowx_put_eggplant_in_basket.yaml
# (fill in the /path/to assets first).
#
# Pick another task with e.g.
#   CONFIG=examples/embodied/groot_n1_7/eval/configs/simplerenv/widowx_carrot_on_plate.yaml \
#   bash examples/embodied/groot_n1_7/eval/run_simplerenv_eval.sh
#
# To reproduce the reported success rates, the policy env named in server.python
# must have transformers 4.57.3 installed; the 5.x series diverges inside the
# Qwen3-VL backbone and drops the six-task WidowX total from 85/120 to 35/120 on
# 5.3.0 (the version our own pyproject.toml installs). For the per-task
# comparison and where the divergence comes from, see
# loongforge/embodied/eval/docs/patches/simplerenv/groot_n1_7.md.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/groot_n1_7/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/simplerenv/widowx_put_eggplant_in_basket.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# libcuda.so.<driver> lives here; without it torch silently falls back to CPU
# and the 3B backbone appears to hang instead of failing.
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIR:-/path/to/nvidia_lib}:/usr/lib64:${LD_LIBRARY_PATH:-}
# SAPIEN renders through Vulkan; point it at the driver ICD.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_lib/10_nvidia.json}

${BENCHMARK_PYTHON:-/path/to/conda/envs/simplerenv/bin/python} -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
