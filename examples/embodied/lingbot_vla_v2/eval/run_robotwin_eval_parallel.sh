#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# LingBot-VLA v2 RoboTwin 2.0 parallel entry: one policy server per GPU, N sim workers per GPU.
# Fill /path/to/... in configs/robotwin/robotwin_50tasks_parallel.yaml first (and scan the offline
# expert-check cache once, see the comments there). Override the config with CONFIG=/path/to/your.yaml.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/lingbot_vla_v2/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/robotwin_50tasks_parallel.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
# Evaluation reads weights from local disk only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# SAPIEN's Vulkan renderer may need an explicit ICD on headless servers.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}
# Do NOT pin CUDA_VISIBLE_DEVICES here: the coordinator starts one policy server per GPU listed under
# `servers[].gpus` and pins each sim worker.

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-$(command -v python3)}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
