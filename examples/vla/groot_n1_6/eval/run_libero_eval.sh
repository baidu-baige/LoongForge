#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.6 LIBERO object task-success.
# Default config: configs/libero/object_smoke.yaml (fill in the /path/to assets).

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/vla/groot_n1_6/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/libero/object_smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIR:-/path/to/nvidia_lib}:/usr/lib64:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
# Route the GR00T Eagle backbone through the repo-local Eagle3 builder (offline,
# from the processor dir's config.json) instead of the HF remote-code loader,
# which would need configuration_eagle3_vl.py inside the processor dir. This env
# only selects the loader; it does NOT trigger cuda-graph capture at inference.
export CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-local}
export CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-full_iteration}

${BENCHMARK_PYTHON:-/path/to/conda/envs/libero/bin/python} -m loongforge.evaluation.orchestrator.run --config "${CONFIG}"
