#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Parallel FastWAM RoboTwin entry: one inference service per GPU, N sim workers
# per GPU, global queue. Fill /path/to/... in
# configs/robotwin/robotwin_fastwam_parallel.yaml first.
# Override the config with CONFIG=/path/to/your.yaml.
#
# The protocol (3-cam T layout, 14D joint actions, replan 24, official step
# limits) is the same as the single-task entry; only the driving changes.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/fastwam/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/robotwin_fastwam_parallel.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/path/to/nvidia_lib:/usr/lib64}
# SAPIEN's Vulkan renderer needs an explicit ICD on headless servers.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_icd.json}
# The service resolves Wan-AI/Wan2.2-TI2V-5B offline under this root.
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/path/to/models}
# Do NOT pin CUDA_VISIBLE_DEVICES: the coordinator starts one service per GPU
# listed under `servers[].gpus` and pins each sim worker to its service.

MODEL_PYTHON=${MODEL_PYTHON:-/path/to/loongforge_vla/bin/python}
"${MODEL_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
