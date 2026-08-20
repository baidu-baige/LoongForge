#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

retention_days="${LOONGFORGE_LOG_RETENTION_DAYS:-7}"
container_retention_hours="${LOONGFORGE_CONTAINER_RETENTION_HOURS:-24}"
runner_log_root="${LOONGFORGE_RUNNER_LOG_ROOT:?LOONGFORGE_RUNNER_LOG_ROOT is required}"
output_root="${LOONGFORGE_HOST_OUTPUT_ROOT:?LOONGFORGE_HOST_OUTPUT_ROOT is required}"
docker_bin="${DOCKER_BIN:-docker}"

[[ "$retention_days" =~ ^[0-9]+$ ]] || {
  echo "LOONGFORGE_LOG_RETENTION_DAYS must be an integer" >&2
  exit 2
}
[[ "$container_retention_hours" =~ ^[0-9]+$ ]] || {
  echo "LOONGFORGE_CONTAINER_RETENTION_HOURS must be an integer" >&2
  exit 2
}

find "$output_root" -mindepth 1 -maxdepth 1 -type d \
  \( -name 'logs_*' -o -name 'run_*' \) \
  -mtime "+$retention_days" -exec rm -rf -- {} +
find "$runner_log_root" -type f -mtime "+$retention_days" -delete

# Only remove stale containers carrying the CI ownership label.
"$docker_bin" ps -aq --filter label=io.loongforge.ci=true --filter status=exited \
  | xargs -r "$docker_bin" rm >/dev/null 2>&1 || true

now_epoch="$(date +%s)"
max_age_seconds=$((container_retention_hours * 3600))
while IFS= read -r container_id; do
  [[ -n "$container_id" ]] || continue
  started_at="$($docker_bin inspect --format '{{.State.StartedAt}}' "$container_id" 2>/dev/null || true)"
  started_epoch="$(date -d "$started_at" +%s 2>/dev/null || true)"
  [[ "$started_epoch" =~ ^[0-9]+$ ]] || continue
  if (( now_epoch - started_epoch > max_age_seconds )); then
    "$docker_bin" rm -f "$container_id" >/dev/null 2>&1 || true
  fi
done < <("$docker_bin" ps -q --filter label=io.loongforge.ci=true)
