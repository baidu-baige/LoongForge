#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

env_code="${1:-}"
case "$env_code" in
  a|p) ;;
  *) echo "unsupported environment" >&2; exit 2 ;;
esac

[[ -d "${SOURCE_DIR:-}" ]] || { echo "SOURCE_DIR is required" >&2; exit 2; }
launcher="${LOONGFORGE_REGRESSION_RUNNER:-}"
if [[ -z "$launcher" ]]; then
  launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/self_runner/run_regression.sh"
fi
[[ -x "$launcher" ]] || { echo "regression runner is not executable: $launcher" >&2; exit 2; }
args=(--source "$SOURCE_DIR" --env "$env_code" --sha "${HEAD_SHA:?HEAD_SHA is required}")
[[ -n "${CI_CONFIG_PROFILE:-}" ]] && args+=(--config-profile "$CI_CONFIG_PROFILE")
# Keep initial CI coverage bounded to the known-good baseline. Maintainers can
# explicitly expand a run with MODELS when another baseline is ready.
models="${MODELS:-deepseek_v2_lite}"
args+=(--model "$models")
[[ -n "${CANDIDATE_REVISION:-}" ]] && args+=(--candidate-revision "$CANDIDATE_REVISION")
exec "$launcher" "${args[@]}"
