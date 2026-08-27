#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Public: GR00T-N1.7 LIBERO libero_10 task-success.
# Default config: configs/libero/libero_10.yaml (fill in the /path/to assets).
#
# To reproduce the reported success rates, the policy env named in server.python
# must have transformers 4.57.3 installed; the 5.x series diverges inside the
# Qwen3-VL backbone and takes libero_10 from 46/50 down to 11/50 on 5.3.0 (the
# version our own pyproject.toml installs). For the per-task comparison
# and where the divergence comes from, see
# loongforge/embodied/eval/docs/patches/libero/groot_n1_7.md.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/groot_n1_7/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/libero/libero_10.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# libcuda.so.<driver> lives here; without it torch silently falls back to CPU
# and the 3B backbone appears to hang instead of failing.
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIR:-/path/to/nvidia_lib}:/usr/lib64:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
# Load the Qwen3-VL / Cosmos-Reason2 backbone and its Qwen3VLProcessor from a
# local dir instead of the gated HF repo. model.model_name in the YAML already
# points there; this env is the override hook used when it does not.
export COSMOS_LOCAL_PATH=${COSMOS_LOCAL_PATH:-/path/to/Cosmos-Reason2-2B}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

${BENCHMARK_PYTHON:-/path/to/conda/envs/libero/bin/python} -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
