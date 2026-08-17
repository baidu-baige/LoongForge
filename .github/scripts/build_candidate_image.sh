#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target="${1:-}"
case "$target" in
  a|h|p) ;;
  *) echo "unsupported image target" >&2; exit 2 ;;
esac

if [[ -n "${CI_CONFIG_PATH:-}" ]]; then
  [[ -f "$CI_CONFIG_PATH" ]] || { echo "CI image config not found" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$CI_CONFIG_PATH"
  set +a
fi

builder="${LOONGFORGE_IMAGE_BUILDER:-}"
[[ -x "$builder" ]] || { echo "LOONGFORGE_IMAGE_BUILDER must point to an executable" >&2; exit 2; }
revision="$($builder \
  --target "$target" \
  --sha "${HEAD_SHA:?HEAD_SHA is required}" \
  --source "${SOURCE_DIR:?SOURCE_DIR is required}")"
[[ "$revision" =~ ^[A-Za-z0-9._:-]+$ ]] || { echo "image builder returned an invalid revision" >&2; exit 2; }
echo "revision=$revision" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
