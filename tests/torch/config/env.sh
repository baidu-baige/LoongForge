#!/bin/bash
# Torch regression centralized path configuration: the only file that needs to be modified per environment.
# Sourced by config/prepare.sh / run.sh;
# all variables are of the form ${VAR:-default}, and can be overridden via environment
# variables before running the entry-point scripts.

_TORCH_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# tests/torch/config -> tests/torch (this suite's self-contained root)
_TORCH_SUITE_ROOT=$(cd "${_TORCH_ENV_DIR}/.." && pwd)

# ── Unified root directory ────────────────────────────────────
# The data and ckpt / logs required for regression are collected under this directory:
#   ${TORCH_CI_ROOT}/
#   ├── vla_artifacts/        # data/ckpt (LOCAL_VLA_ARTIFACTS_ROOT)
#   ├── logs/                 # regression logs (TORCH_LOG_ROOT)
#   └── tools/                # optional artifact preparation tools
export TORCH_CI_ROOT=${TORCH_CI_ROOT:-"/workspace/loongforge_torch_ci"}

# ── Data and artifacts root directory ─────────────────────────
# Training reads LOCAL_VLA_ARTIFACTS_ROOT following the <family>/{models,datasets,tokenizers} structure,
# and the default ckpt/data paths of the examples/{vla,world} training scripts are also derived from it.
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"${TORCH_CI_ROOT}/vla_artifacts"}

# ── Regression log/result root directory (read by cli.py) ──
export TORCH_LOG_ROOT=${TORCH_LOG_ROOT:-"${TORCH_CI_ROOT}/logs"}

# ── baseline root directory (baseline/<chip>/<model>.json) ──
# Defaults in-repo under tests/torch/baseline, keeping this suite self-contained
# (the mcore suite owns tests/mcore/baseline/{default,optional} separately).
# Override with TORCH_BASELINE_ROOT to point at a shared out-of-repo collection
# (e.g. when running the same checkout from multiple machines).
export TORCH_BASELINE_ROOT=${TORCH_BASELINE_ROOT:-"${_TORCH_SUITE_ROOT}/baseline"}

# ── Artifact source and optional preparation tool ─────────────
export BOS_VLA_ARTIFACTS_ROOT=${BOS_VLA_ARTIFACTS_ROOT:-"bos:/path/to/vla_artifacts/"}
export BCECMD_DIR=${BCECMD_DIR:-"${TORCH_CI_ROOT}/tools"}
export BCECMD=${BCECMD:-"${BCECMD_DIR}/bcecmd"}
