#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public open-source entry for RoboTwin (pi0.5 task-success template).
# Fill /path/to/... in configs/robotwin/adjust_bottle_smoke.yaml first.
# Optional task/bridge/random_init knobs: see comments in that YAML.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/adjust_bottle_smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/path/to/nvidia_lib:/usr/lib64}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_icd.json}

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/robotwin/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
