#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public open-source entry: fill /path/to/... in the YAML first.
# Default: configs/robodojo/stack_bowls_smoke.yaml (one task, layout seed 0, 2 episodes).
# Full protocol: CONFIG=examples/embodied/lingbot_va/eval/configs/robodojo/robodojo_all_tasks.yaml
#
# The orchestrator itself only needs PyYAML and the LoongForge eval package; the
# policy server runs under server.python and RoboDojo's sim client under
# env.robodojo_python (Isaac Sim 5.1 environment).

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/lingbot_va/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robodojo/stack_bowls_smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Checkpoints and assets are read from local disk only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ORCHESTRATOR_PYTHON=${ORCHESTRATOR_PYTHON:-python3}
"${ORCHESTRATOR_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
