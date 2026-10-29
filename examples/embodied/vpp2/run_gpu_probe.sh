#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
export VPP2_BASE_ROOT=${VPP2_BASE_ROOT:?Set official checkout path}
export VPP2_LOONG_ROOT=${VPP2_LOONG_ROOT:-$ROOT}
export VPP2_ASSETS_ROOT=${VPP2_ASSETS_ROOT:?Set real assets directory}
export VPP2_ALIGNMENT_INIT=${VPP2_ALIGNMENT_INIT:?Use same init file on both sides}
export PYTHONPATH="$ROOT:$VPP2_BASE_ROOT/src:$VPP2_BASE_ROOT/scripts/repro:${PYTHONPATH:-}"
exec "${PYTHON_BIN:-python}" "$HERE/gpu_full_model_probe.py" --side loongforge "$@"
