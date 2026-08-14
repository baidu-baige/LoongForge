#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public open-source entry: fill /path/to/... in the YAML first.
# Default (only shipped config): configs/libero/libero_10.yaml
# Optional suite/episode knobs: see comments in that YAML.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/lingbot_va/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/libero/libero_10.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/libero/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
