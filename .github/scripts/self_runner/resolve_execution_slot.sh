#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Resolve a public Execution Slot to an operator-managed execution target.
# The configuration file contains internal topology and is never committed.

set -euo pipefail

mask_value() {
  local value="${1:-}"
  if [[ -n "$value" && "${#value}" -ge 4 ]]; then
    printf '::add-mask::%s\n' "$value"
  fi
}

slot="${1:-${LOONGFORGE_RELEASE_EXECUTION_SLOT:-}}"
[[ "$slot" =~ ^(primary|secondary)$ ]] || {
  printf '%s\n' 'execution-slot: invalid' >&2
  exit 2
}

config="${LOONGFORGE_EXECUTION_SLOT_CONFIG:-}"
[[ -n "$config" ]] || { printf '%s\n' 'execution-slot: unavailable' >&2; exit 2; }
[[ -f "$config" && ! -L "$config" ]] || {
  printf '%s\n' 'execution-slot: unavailable' >&2
  exit 2
}
mode="$(stat -c '%a' "$config" 2>/dev/null || true)"
[[ "$mode" =~ ^[0-7]+$ ]] && (( (8#$mode & 077) == 0 )) || {
  printf '%s\n' 'execution-slot: insecure configuration' >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "$config"
set +a

prefix="LOONGFORGE_EXECUTION_SLOT_${slot^^}"
mode_var="${prefix}_MODE"
execution_mode="${!mode_var:-}"
[[ "$execution_mode" =~ ^(local|ssh)$ ]] || {
  printf '%s\n' 'execution-slot: incomplete' >&2
  exit 2
}

alias_var="${prefix}_ALIAS"
docker_host_var="${prefix}_DOCKER_HOST"
root_var="${prefix}_ROOT"
source_var="${prefix}_SOURCE"
config_var="${prefix}_CONFIG"
execution_alias="${!alias_var:-}"
docker_host="${!docker_host_var:-}"
execution_root="${!root_var:-}"
execution_source="${!source_var:-}"
execution_config="${!config_var:-}"

if [[ "$execution_mode" == ssh ]]; then
  for name in execution_alias docker_host execution_root execution_source execution_config; do
    [[ -n "${!name}" ]] || {
      printf '%s\n' 'execution-slot: incomplete' >&2
      exit 2
    }
  done
  [[ "$docker_host" =~ ^ssh://[^[:space:]]+$ ]] || {
    printf '%s\n' 'execution-slot: invalid configuration' >&2
    exit 2
  }
else
  execution_alias=""
  execution_root=""
  execution_source=""
  execution_config=""
fi

export LOONGFORGE_RELEASE_EXECUTION_SLOT="$slot"
export LOONGFORGE_EXECUTION_MODE="$execution_mode"
export LOONGFORGE_EXECUTION_ALIAS="$execution_alias"
export LOONGFORGE_EXECUTION_ROOT="$execution_root"
export LOONGFORGE_EXECUTION_SOURCE="$execution_source"
export LOONGFORGE_EXECUTION_CONFIG="$execution_config"
export DOCKER_HOST="$docker_host"

for value in \
  "$execution_alias" \
  "$execution_root" \
  "$execution_source" \
  "$execution_config" \
  "$docker_host"; do
  mask_value "$value"
done

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s=%s\n' LOONGFORGE_RELEASE_EXECUTION_SLOT "$slot" >>"$GITHUB_ENV"
  printf '%s=%s\n' LOONGFORGE_EXECUTION_MODE "$execution_mode" >>"$GITHUB_ENV"
  printf '%s=%s\n' LOONGFORGE_EXECUTION_ALIAS "$execution_alias" >>"$GITHUB_ENV"
  printf '%s=%s\n' LOONGFORGE_EXECUTION_ROOT "$execution_root" >>"$GITHUB_ENV"
  printf '%s=%s\n' LOONGFORGE_EXECUTION_SOURCE "$execution_source" >>"$GITHUB_ENV"
  printf '%s=%s\n' LOONGFORGE_EXECUTION_CONFIG "$execution_config" >>"$GITHUB_ENV"
  printf '%s=%s\n' DOCKER_HOST "$docker_host" >>"$GITHUB_ENV"
fi
