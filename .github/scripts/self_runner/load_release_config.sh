#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Sourced (not executed) by release workflow steps. Loads only operator-managed
# release configuration; credentials are loaded by the login helper on demand.

config="${LOONGFORGE_RELEASE_CONFIG:-}"
[[ -n "$config" ]] || { printf '%s\n' 'release-config: unavailable' >&2; return 2; }
[[ -f "$config" && ! -L "$config" ]] || {
  printf '%s\n' 'release-config: unavailable' >&2
  return 2
}
config_mode="$(stat -c '%a' "$config" 2>/dev/null || true)"
[[ "$config_mode" =~ ^[0-7]+$ ]] && (( (8#$config_mode & 077) == 0 )) || {
  printf '%s\n' 'release-config: insecure configuration' >&2
  return 2
}

set -a
# shellcheck disable=SC1090
source "$config"
set +a

required=(
  DOCKER_HOST
  LOONGFORGE_RELEASE_PUBLISHER_ALIAS
  LOONGFORGE_DOCKERHUB_PROXY
  LOONGFORGE_DOCKERHUB_NO_PROXY
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf '%s\n' 'release-config: incomplete' >&2
    return 2
  }
done

for path in "$LOONGFORGE_IREGISTRY_CREDENTIALS" "$LOONGFORGE_DOCKERHUB_CREDENTIALS"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    printf '%s\n' 'release-config: unavailable' >&2
    return 2
  }
  mode="$(stat -c '%a' "$path" 2>/dev/null || true)"
  [[ "$mode" =~ ^[0-7]+$ ]] && (( (8#$mode & 077) == 0 )) || {
    printf '%s\n' 'release-config: insecure credentials' >&2
    return 2
  }
done
