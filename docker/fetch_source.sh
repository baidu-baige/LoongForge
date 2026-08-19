#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: fetch_source.sh <manifest-key> <destination> <archive-root> <fallback-command...>" >&2
  exit 2
fi

manifest_key="$1"
destination="$2"
archive_root="$3"
shift 3

manifest="/run/secrets/source_manifest"
if [[ ! -s "$manifest" ]]; then
  "$@"
  printf 'public\n'
  exit 0
fi

url_var="${manifest_key}_URL"
sha_var="${manifest_key}_SHA256"
had_xtrace=false
[[ $- == *x* ]] && had_xtrace=true
set +x
# The manifest is generated and owned by the runner operator.
# shellcheck disable=SC1090
source "$manifest"
url="${!url_var:-}"
expected_sha="${!sha_var:-}"
[[ -n "$url" && "$url" == http* && "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo "missing or invalid internal source entry: $manifest_key" >&2
  exit 1
}

archive="$(mktemp)"
trap 'rm -f "$archive"' EXIT
wget --quiet --timeout=120 --tries=3 -O "$archive" "$url"
printf '%s  %s\n' "$expected_sha" "$archive" | sha256sum --check --status
[[ "$had_xtrace" == true ]] && set -x

parent="$(dirname "$destination")"
mkdir -p "$parent"
tar -xzf "$archive" -C "$parent"
extracted="$parent/$archive_root"
[[ -d "$extracted" ]] || {
  echo "archive for $manifest_key does not contain $archive_root" >&2
  exit 1
}
if [[ "$extracted" != "$destination" ]]; then
  [[ ! -e "$destination" ]] || {
    echo "source destination already exists: $destination" >&2
    exit 1
  }
  mv "$extracted" "$destination"
fi
printf 'internal\n'
