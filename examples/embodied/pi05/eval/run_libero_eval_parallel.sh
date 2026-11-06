#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Parallel LIBERO entry: one inference service per GPU, N sim workers per GPU.
# Fill /path/to/... in configs/libero/libero_pi05_parallel.yaml first.
# Override the config with CONFIG=/path/to/your.yaml.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/libero/libero_pi05_parallel.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
# Do NOT pin CUDA_VISIBLE_DEVICES here: the coordinator starts one inference
# service per GPU listed under `servers[].gpus` and pins each sim worker.
# Leave it unset (or list every GPU) when more than one card is used.

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/libero/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
