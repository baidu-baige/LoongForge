#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Register private runner values with GitHub's log masker before any action or
# shell command that could echo the execution environment.

set -euo pipefail

mask() {
  if [[ -n "${1:-}" && "${#1}" -ge 4 ]]; then
    printf '::add-mask::%s\n' "$1"
  fi
}

for name in RUNNER_NAME RUNNER_TEMP RUNNER_TOOL_CACHE GITHUB_WORKSPACE \
            CI_CONFIG_PATH CI_CONFIG_PATH_IMAGE LOONGFORGE_RELEASE_CONFIG \
            LOONGFORGE_EXECUTION_SLOT_CONFIG \
            LOONGFORGE_IREGISTRY_CREDENTIALS LOONGFORGE_DOCKERHUB_CREDENTIALS \
            INTERNAL_IMAGE_REPOSITORY DOCKERHUB_IMAGE DOCKER_HOST DOCKER_CONFIG \
            HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  mask "${!name:-}"
done
mask "$(hostname 2>/dev/null || true)"

for config in "${CI_CONFIG_PATH_IMAGE:-}" "${LOONGFORGE_RELEASE_CONFIG:-}" \
              "${LOONGFORGE_EXECUTION_SLOT_CONFIG:-}" \
              "${LOONGFORGE_IREGISTRY_CREDENTIALS:-}" \
              "${LOONGFORGE_DOCKERHUB_CREDENTIALS:-}"; do
  [[ -f "$config" && ! -L "$config" ]] || continue
  while IFS='=' read -r name value; do
    if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      mask "$value"
    fi
  done < "$config"
done

mask_values="${LOONGFORGE_RUNNER_MASK_VALUES:-}"
if [[ -n "$mask_values" && -f "$mask_values" && ! -L "$mask_values" ]]; then
  while IFS= read -r value; do
    if [[ -n "$value" && "$value" != \#* ]]; then
      mask "$value"
    fi
  done < "$mask_values"
fi
