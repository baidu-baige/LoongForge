#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public open-source entry: FastWAM LIBERO task-success eval.
# One config per suite: configs/libero/libero_{spatial,object,goal,10}.yaml.
# Fill /path/to/... in your suite's yaml first, then point CONFIG at it
# (default: libero_spatial.yaml).
# Open weights: libero_optional_idm_2cam224.pt + Wan-AI/Wan2.2-TI2V-5B
# (links in the YAML). Optional suite/episode/mode knobs: see the YAML.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/fastwam/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/libero/libero_spatial.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/path/to/nvidia_lib:/usr/lib64}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
# Offline Wan release lookup (the model server inherits this env). Point at
# the dir that CONTAINS Wan-AI/Wan2.2-TI2V-5B, i.e. ${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B
# resolves to the release dir with T5/VAE + google/umt5-xxl inside.
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/path/to/models_root}

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/libero/bin/python}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
