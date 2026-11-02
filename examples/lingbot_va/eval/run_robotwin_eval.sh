#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public open-source entry: fill /path/to/... in the YAML first.
# Default: configs/robotwin/robotwin_50tasks.yaml, the full 50-task protocol of
# the released numbers (Easy; set benchmark.task_config: demo_randomized for Hard).
# Smoke test: CONFIG=examples/lingbot_va/eval/configs/robotwin/adjust_bottle.yaml

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/lingbot_va/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/robotwin_50tasks.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/robotwin/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.evaluation.embodied.orchestrator.run --config "${CONFIG}"
