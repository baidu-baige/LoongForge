#!/bin/bash
# Embodied regression centralized path configuration: the only file that needs to be modified per environment.
# Sourced by config/prepare.sh / run.sh;
# all variables are of the form ${VAR:-default}, and can be overridden via environment
# variables before running the entry-point scripts.

_EMBODIED_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# tests/embodied/config -> tests/embodied (this suite's self-contained root)
_EMBODIED_SUITE_ROOT=$(cd "${_EMBODIED_ENV_DIR}/.." && pwd)

# ── Unified root directory ────────────────────────────────────
# The data and ckpt / logs / tools required for regression are all collected under this directory:
#   ${EMBODIED_CI_ROOT}/
#   ├── vla_artifacts/        # data/ckpt (LOCAL_VLA_ARTIFACTS_ROOT)
#   ├── logs/                 # regression logs (EMBODIED_LOG_ROOT)
#   └── tools/                # tools such as bcecmd (BCECMD_DIR)
export EMBODIED_CI_ROOT=${EMBODIED_CI_ROOT:-"/ssd2/loongforge_embodied_ci"}

# ── Data and artifacts root directory ─────────────────────────
# Download and training share the same root: config/prepare.sh downloads to
# LOCAL_VLA_ARTIFACTS_ROOT following the <family>/{models,datasets,tokenizers} structure,
# and the default ckpt/data paths of the examples/embodied training scripts are also derived from it.
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"${EMBODIED_CI_ROOT}/vla_artifacts"}

# ── Regression log/result root directory (read by cli.py) ──
export EMBODIED_LOG_ROOT=${EMBODIED_LOG_ROOT:-"${EMBODIED_CI_ROOT}/logs"}

# ── baseline root directory (baseline/<chip>/<model>.json) ──
# Defaults in-repo under tests/embodied/baseline, keeping this suite self-contained
# (the llm_vlm suite owns tests/llm_vlm/baseline/{default,optional} separately).
# Override with EMBODIED_BASELINE_ROOT to point at a shared out-of-repo collection
# (e.g. when running the same checkout from multiple machines).
export EMBODIED_BASELINE_ROOT=${EMBODIED_BASELINE_ROOT:-"${_EMBODIED_SUITE_ROOT}/baseline"}

# ── BOS source prefix (used by config/prepare.sh) ─────────────
export BOS_VLA_ARTIFACTS_ROOT=${BOS_VLA_ARTIFACTS_ROOT:-"bos:/path/to/vla_artifacts/"}

# ── Tools ─────────────────────────────────────────────────────
# The directory where bcecmd resides; you can also specify the full path directly with BCECMD
export BCECMD_DIR=${BCECMD_DIR:-"${EMBODIED_CI_ROOT}/tools"}
export BCECMD=${BCECMD:-"${BCECMD_DIR}/bcecmd"}

