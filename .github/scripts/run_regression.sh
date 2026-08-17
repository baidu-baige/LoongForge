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
if [[ -n "${CI_CONFIG_PATH:-}" ]]; then
  [[ -f "$CI_CONFIG_PATH" ]] || { echo "CI regression config not found" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$CI_CONFIG_PATH"
  set +a
fi

launcher="${LOONGFORGE_REGRESSION_RUNNER:-}"
if [[ -n "$launcher" ]]; then
  [[ -x "$launcher" ]] || { echo "LOONGFORGE_REGRESSION_RUNNER is not executable" >&2; exit 2; }
  args=(--source "$SOURCE_DIR" --env "$env_code" --sha "${HEAD_SHA:?HEAD_SHA is required}")
  [[ -n "${MODELS:-}" ]] && args+=(--model "$MODELS")
  [[ -n "${CANDIDATE_REVISION:-}" ]] && args+=(--candidate-revision "$CANDIDATE_REVISION")
  exec "$launcher" "${args[@]}"
fi

if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
  echo "LOONGFORGE_REGRESSION_RUNNER is required on self-hosted CI" >&2
  exit 2
fi

args=(--env "$env_code")
[[ -n "${MODELS:-}" ]] && args+=(--model "$MODELS")
exec bash "$SOURCE_DIR/tests/main_start.sh" "${args[@]}"
