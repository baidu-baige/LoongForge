#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# LingBot-VLA v2 RoboTwin 2.0 eval entry.
# Default config: configs/robotwin/adjust_bottle_smoke.yaml; edit its /path/to entries first
# (server.ckpt_path / python, model.qwen3vl_path, env.robotwin_root / robotwin_python, run.output_dir).
# Full benchmark: CONFIG=examples/embodied/lingbot_vla_v2/eval/configs/robotwin/robotwin_50tasks.yaml
#
# Env (all optional, config wins):
#   REPO_ROOT          LoongForge checkout (default /path/to/LoongForge)
#   CONFIG             config override (absolute or REPO_ROOT-relative)
#   CUDA_VISIBLE_DEVICES  GPU for the policy server and the simulator
#   BENCHMARK_PYTHON   python that runs the orchestrator (any env with pyyaml/omegaconf)

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/path/to/LoongForge}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/lingbot_vla_v2/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/robotwin/adjust_bottle_smoke.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Evaluation reads weights from local disk only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# SAPIEN's Vulkan renderer may need an explicit ICD on headless servers.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}

BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-$(command -v python3)}
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
