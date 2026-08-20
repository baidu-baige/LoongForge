#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

pip_config="/run/secrets/pip_config"
had_xtrace=false
[[ $- == *x* ]] && had_xtrace=true
[[ "$had_xtrace" == true ]] && set +x

if [[ -s "$pip_config" ]]; then
  export PIP_CONFIG_FILE="$pip_config"
  uv_index_url="$(awk -F= '/^[[:space:]]*index-url[[:space:]]*=/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$pip_config")"
  if [[ -z "$uv_index_url" ]]; then
    echo "pip config does not define index-url" >&2
    [[ "$had_xtrace" == true ]] && set -x
    return 1
  fi
  export UV_INDEX_URL="$uv_index_url"
fi

[[ "$had_xtrace" == true ]] && set -x
