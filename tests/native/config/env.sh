#!/bin/bash
# Native regression centralized path configuration: the only file that needs to be modified per environment.
# Sourced by config/prepare.sh / run.sh;
# all variables are of the form ${VAR:-default}, and can be overridden via environment
# variables before running the entry-point scripts.

_NATIVE_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# tests/native/config -> tests/native (this suite's self-contained root)
_NATIVE_SUITE_ROOT=$(cd "${_NATIVE_ENV_DIR}/.." && pwd)

# ── Unified root directory ────────────────────────────────────
# The data and ckpt / logs required for regression are collected under this directory:
#   ${NATIVE_CI_ROOT}/
#   ├── vla_artifacts/        # data/ckpt (LOCAL_VLA_ARTIFACTS_ROOT)
#   ├── logs/                 # regression logs (NATIVE_LOG_ROOT)
#   └── tools/                # optional artifact preparation tools
export NATIVE_CI_ROOT=${NATIVE_CI_ROOT:-"/workspace/loongforge_native_ci"}

# ── Data and artifacts root directory ─────────────────────────
# Training reads LOCAL_VLA_ARTIFACTS_ROOT following the <family>/{models,datasets,tokenizers} structure,
# and the default ckpt/data paths of the examples/{vla,world} training scripts are also derived from it.
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"${NATIVE_CI_ROOT}/vla_artifacts"}

# ── Regression log/result root directory (read by cli.py) ──
export NATIVE_LOG_ROOT=${NATIVE_LOG_ROOT:-"${NATIVE_CI_ROOT}/logs"}

# ── baseline root directory (baseline/<chip>/<model>.json) ──
# Defaults in-repo under tests/native/baseline, keeping this suite self-contained
# (the llm_vlm suite owns tests/llm_vlm/baseline/{default,optional} separately).
# Override with NATIVE_BASELINE_ROOT to point at a shared out-of-repo collection
# (e.g. when running the same checkout from multiple machines).
export NATIVE_BASELINE_ROOT=${NATIVE_BASELINE_ROOT:-"${_NATIVE_SUITE_ROOT}/baseline"}

# ── Artifact source and optional preparation tool ─────────────
export BOS_VLA_ARTIFACTS_ROOT=${BOS_VLA_ARTIFACTS_ROOT:-"bos:/path/to/vla_artifacts/"}
export BCECMD_DIR=${BCECMD_DIR:-"${NATIVE_CI_ROOT}/tools"}
export BCECMD=${BCECMD:-"${BCECMD_DIR}/bcecmd"}
