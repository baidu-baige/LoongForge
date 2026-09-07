#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# FastWAM RoboTwin task-success eval entry (open source).
# Default config: configs/robotwin/adjust_bottle.yaml — edit its paths
# (server.ckpt_path / dataset_statistics_path, env.robotwin_root / robotwin_python,
# model.tokenizer_model_id, server.tokenizer_path, run.output_dir, ...) to your
# machine before running.
#
# Weight: official FastWAM robotwin_uncond_3cam_384.pt
#   https://huggingface.co/yuantianyuan01/FastWAM-robotwin (or the repo release)
# RoboTwin release: https://github.com/TianxingChen/RoboTwin
#
# Env (all optional, config wins):
#   REPO_ROOT   LoongForge-VLA checkout (default /path/to/LoongForge-VLA)
#   CONFIG      config override (absolute or REPO_ROOT-relative)
#   CUDA_VISIBLE_DEVICES  model-server GPU
#   NVIDIA_LIB_DIR  dir with CUDA user-space libs (libnvJitLink etc.)

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge-VLA}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/fastwam/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/adjust_bottle.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

NVIDIA_LIB_DIR=${NVIDIA_LIB_DIR:-/path/to/nvidia_lib}

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=${NVIDIA_LIB_DIR}:/usr/lib64:${LD_LIBRARY_PATH:-}
# SAPIEN's Vulkan renderer may need an explicit ICD (headless servers without
# the glvnd ICD in the default search path); without it the official evaluator
# dies with "Render Error" before the first episode.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}

# The model server resolves Wan-AI/Wan2.2-TI2V-5B offline under this root
# (config sets redirect_common_files: false); override if your release lives elsewhere.
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$(dirname "${REPO_ROOT}")}

MODEL_PYTHON=${MODEL_PYTHON:-$(command -v python3)}
"${MODEL_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
