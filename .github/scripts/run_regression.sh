#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

suite="${1:-}"
case "$suite" in
  llm_vlm|embodied) ;;
  *) echo "unsupported test suite" >&2; exit 2 ;;
esac

[[ -d "${SOURCE_DIR:-}" ]] || { echo "SOURCE_DIR is required" >&2; exit 2; }

config="${CI_CONFIG_PATH_IMAGE:-${CI_CONFIG_PATH:-}}"
if [[ -n "$config" ]]; then
  [[ -f "$config" ]] || { echo "CI image config not found" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$config"
  set +a
fi

launcher="${LOONGFORGE_REGRESSION_RUNNER:-}"
if [[ -z "$launcher" ]]; then
  launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/self_runner/run_regression.sh"
fi
[[ -x "$launcher" ]] || { echo "regression runner is not executable: $launcher" >&2; exit 2; }
args=(--source "$SOURCE_DIR" --suite "$suite" --sha "${HEAD_SHA:?HEAD_SHA is required}")
if [[ "$suite" == embodied ]]; then
  models="${MODELS:-pi05_ddp}"
else
  models="${MODELS:-deepseek_v2_lite}"
fi
args+=(--model "$models")
[[ -n "${CANDIDATE_REVISION:-}" ]] && args+=(--candidate-revision "$CANDIDATE_REVISION")
exec "$launcher" "${args[@]}"
