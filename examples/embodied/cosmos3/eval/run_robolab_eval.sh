#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# RoboLab (Isaac Sim) evaluation of Cosmos3 with the LoongForge policy server.
#
# Defaults to CONFIG=configs/robolab/cosmos3.yaml: 120 tasks x 10 episodes, 6 policy
# servers (GPU0-2) + 6 Isaac Sim workers (GPU3-7). Replace the /path/to placeholders there
# first. The simulator runs inside the benchmark python, so BENCHMARK_PYTHON must be the
# RoboLab virtualenv (isaacsim + isaaclab + robolab), not the LoongForge training env.
# GPU placement comes from the YAML (servers[].gpus / pools.gpu_sim); CUDA_VISIBLE_DEVICES
# is only used for single-GPU smoke configs.
#
# The container-level Isaac Sim fixes are inherited from this shell:
#   LD_PRELOAD=<glapi shim>  LD_LIBRARY_PATH=<real libcuda>

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/cosmos3/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robolab/cosmos3.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

# Container / headless fixes. Override any of them from the shell before calling this script.
# A proxy MUST be set (even one that fails fast, e.g. http://127.0.0.1:9): with no proxy at
# all Isaac Sim stalls 18+ min in physx/usdrt scene filling. localhost must bypass it or the
# policy websocket goes through the proxy.
export http_proxy=${http_proxy:-http://127.0.0.1:9}
export https_proxy=${https_proxy:-${http_proxy}}
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${HOME}/.cache/robolab}
export TMPDIR=${TMPDIR:-${XDG_CACHE_HOME}/tmp}
mkdir -p "${TMPDIR}"
export OMNI_KIT_ACCEPT_EULA=Y
# Only needed when the container's glvnd/libglapi versions disagree; leave unset otherwise.
#   export LD_PRELOAD=<path>/libglapi_shim.so

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/RoboLab/.venv/bin/python}
set +e
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
status=$?
set -e

# Auto-summary: group success rates + leaderboard comparison once the run is valid.
# ROBOLAB_ROOT is only needed for result files written before episodes carried task tags.
if [[ -z "${ROBOLAB_ROOT:-}" ]]; then
  ROBOLAB_ROOT=$("${BENCHMARK_PYTHON}" -c \
    "import os, robolab; print(os.path.dirname(os.path.dirname(robolab.__file__)))" 2>/dev/null || true)
fi
export ROBOLAB_ROOT
OUTPUT_DIR=$(sed -n 's/^  output_dir: *//p' "${CONFIG}" | head -n 1)
RESULT=${OUTPUT_DIR}/coordinator_result.json
if [[ -f "${RESULT}" ]] && python3 -c "
import json,sys
v=json.load(open('${RESULT}'))['validation']
sys.exit(0 if v.get('ok') else 1)" 2>/dev/null; then
  MODEL_TYPE=$(sed -n 's/^  model_type: *//p' "${CONFIG}" | head -n 1)
  [[ "${MODEL_TYPE}" == pi05 ]] && LB=pi05 || LB=cosmos3
  echo
  python3 -m loongforge.embodied.eval.metrics.robolab_report "${OUTPUT_DIR}" "${LB}" || true
elif [[ ${status} -eq 0 ]]; then
  echo "[robolab] no valid ${RESULT}; skipping the summary" >&2
fi
exit "${status}"
