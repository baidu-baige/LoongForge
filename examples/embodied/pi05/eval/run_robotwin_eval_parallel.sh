#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Parallel RoboTwin entry: one inference service per GPU, N sim workers per GPU.
# Fill /path/to/... in configs/robotwin/robotwin_pi05_parallel.yaml first.
# Override the config with CONFIG=/path/to/your.yaml.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/robotwin_pi05_parallel.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/path/to/nvidia_lib:/usr/lib64}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_icd.json}
# Do NOT pin CUDA_VISIBLE_DEVICES here: the coordinator starts one inference
# service per GPU listed under `servers[].gpus` and pins each sim worker.

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/robotwin/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
