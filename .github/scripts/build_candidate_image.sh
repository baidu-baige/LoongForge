#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target="${1:-}"
case "$target" in
  a|p) ;;
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
builder="${LOONGFORGE_IMAGE_BUILDER:-$script_dir/self_runner/build_candidate_image.sh}"
[[ -x "$builder" ]] || { echo "LOONGFORGE_IMAGE_BUILDER must point to an executable" >&2; exit 2; }
revision="$("$builder" \
  --target "$target" \
  --sha "${HEAD_SHA:?HEAD_SHA is required}" \
  --source "${SOURCE_DIR:?SOURCE_DIR is required}" \
  --pr "${PR_NUMBER:?PR_NUMBER is required}" \
  --tree-sha "${TREE_SHA:?TREE_SHA is required}")"
[[ "$revision" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "image builder returned an invalid revision" >&2; exit 2; }
echo "revision=$revision" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
