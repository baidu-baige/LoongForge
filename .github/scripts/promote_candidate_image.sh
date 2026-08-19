#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target="${1:-all}"
case "$target" in
  a|h|p|all) ;;
  *) echo "unsupported image target" >&2; exit 2 ;;
esac

config="${CI_CONFIG_PATH:-${CI_CONFIG_PATH_IMAGE:-}}"
if [[ -n "$config" ]]; then
  [[ -f "$config" ]] || { echo "CI image config not found: $config" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$config"
  set +a
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
promoter="${LOONGFORGE_IMAGE_PROMOTER:-$script_dir/self_runner/promote_candidate_image.sh}"
[[ -x "$promoter" ]] || { echo "LOONGFORGE_IMAGE_PROMOTER must point to an executable" >&2; exit 2; }
exec "$promoter" \
  --target "$target" \
  --pr "${PR_NUMBER:?PR_NUMBER is required}" \
  --head-sha "${HEAD_SHA:?HEAD_SHA is required}" \
  --merge-sha "${MERGE_SHA:?MERGE_SHA is required}" \
  --tree-sha "${TREE_SHA:?TREE_SHA is required}"
