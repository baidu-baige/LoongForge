#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

suite="${1:-}"
case "$suite" in
  llm_vlm|native) ;;
  *) echo "unsupported test suite" >&2; exit 2 ;;
esac

[[ -d "${SOURCE_DIR:-}" ]] || { echo "SOURCE_DIR is required" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/load_ci_config.sh"

launcher="${LOONGFORGE_REGRESSION_RUNNER:-}"
if [[ -z "$launcher" ]]; then
  launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/self_runner/run_regression.sh"
fi
[[ -f "$launcher" && -x "$launcher" ]] || {
  echo "regression runner is not executable" >&2
  exit 2
}
args=(--source "$SOURCE_DIR" --suite "$suite" --sha "${HEAD_SHA:?HEAD_SHA is required}")
if [[ "$suite" == native ]]; then
  models="${MODELS:-pi05_ddp}"
else
  models="${MODELS:-deepseek_v2_lite}"
fi
args+=(--model "$models")
[[ -n "${CANDIDATE_REVISION:-}" ]] && args+=(--candidate-revision "$CANDIDATE_REVISION")
exec "$launcher" "${args[@]}"
