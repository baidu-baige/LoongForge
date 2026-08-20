#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target="${1:-all}"
case "$target" in
  a|h|p|all) ;;
  *) echo "unsupported image target" >&2; exit 2 ;;
esac

if [[ -n "${CI_CONFIG_PATH:-}" ]]; then
  [[ -f "$CI_CONFIG_PATH" ]] || { echo "CI image config not found" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$CI_CONFIG_PATH"
  set +a
fi

promoter="${LOONGFORGE_IMAGE_PROMOTER:-}"
[[ -x "$promoter" ]] || { echo "LOONGFORGE_IMAGE_PROMOTER must point to an executable" >&2; exit 2; }
exec "$promoter" \
  --target "$target" \
  --pr "${PR_NUMBER:?PR_NUMBER is required}" \
  --head-sha "${HEAD_SHA:?HEAD_SHA is required}" \
  --merge-sha "${MERGE_SHA:?MERGE_SHA is required}"
