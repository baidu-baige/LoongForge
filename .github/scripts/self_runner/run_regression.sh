#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=""
env_code=""
profile=""
sha=""
models=""
candidate_revision=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_dir="${2:-}"; shift 2 ;;
    --env) env_code="${2:-}"; shift 2 ;;
    --config-profile) profile="${2:-}"; shift 2 ;;
    --sha) sha="${2:-}"; shift 2 ;;
    --model) models="${2:-}"; shift 2 ;;
    --candidate-revision) candidate_revision="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$env_code" =~ ^(a|p)$ ]] || { echo "environment must be a or p" >&2; exit 2; }
[[ -d "$source_dir" && -n "$sha" ]] || { echo "source and sha are required" >&2; exit 2; }
[[ -z "$profile" || "$profile" == "$env_code" ]] || { echo "profile and environment differ" >&2; exit 2; }

: "${LOONGFORGE_DEFAULT_IMAGE:?LOONGFORGE_DEFAULT_IMAGE is required}"
: "${LOONGFORGE_PFS_ROOT:?LOONGFORGE_PFS_ROOT is required}"
: "${LOONGFORGE_CONTAINER_PFS:?LOONGFORGE_CONTAINER_PFS is required}"
: "${LOONGFORGE_CONTAINER_SOURCE:?LOONGFORGE_CONTAINER_SOURCE is required}"
: "${LOONGFORGE_RUNNER_LOG_ROOT:?LOONGFORGE_RUNNER_LOG_ROOT is required}"
docker_bin="${DOCKER_BIN:-docker}"
mkdir -p "$LOONGFORGE_RUNNER_LOG_ROOT"

image="${candidate_revision:-$LOONGFORGE_DEFAULT_IMAGE}"
container_name="loongforge-ci-${env_code}-${sha:0:12}-$$"
log_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.log"
result_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.result.json"
resume_state_file="$LOONGFORGE_CONTAINER_PFS/logs/resume/${container_name}.json"
mkdir -p "$LOONGFORGE_PFS_ROOT/logs/resume"

status=1
outputs_written=false
write_outputs() {
  if [[ "$outputs_written" == false && -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'result_json=%s\nlog_file=%s\n' "$result_file" "$log_file" >>"$GITHUB_OUTPUT"
    outputs_written=true
  fi
}
cleanup() {
  "$docker_bin" logs "$container_name" >>"$log_file" 2>&1 || true
  "$docker_bin" rm -f "$container_name" >/dev/null 2>&1 || true
  if [[ ! -f "$result_file" ]]; then
    printf '{"status":"failed","environment":"%s","sha":"%s","models":"%s","exit_code":%d,"log":"%s"}\n' \
      "$env_code" "$sha" "$models" "$status" "$log_file" >"$result_file"
  fi
  "$script_dir/cleanup.sh" >>"$log_file" 2>&1 || true
  write_outputs
  exit "$status"
}
trap cleanup EXIT INT TERM

"$script_dir/create_container.sh" "$image" "$source_dir" "$env_code" "$container_name" >>"$log_file" 2>&1

exec_args=(--env "$env_code")
[[ -n "$models" ]] && exec_args+=(--models "$models")
set +e
"$docker_bin" exec \
  -e "PFS_PATH=$LOONGFORGE_CONTAINER_PFS" \
  -e "TRAINING_LOG_PATH=$LOONGFORGE_CONTAINER_PFS/logs" \
  -e "LOONGFORGE_TEST_ENV=$env_code" \
  -e "RESUME_STATE_FILE=$resume_state_file" \
  "$container_name" \
  bash -lc "cd '$LOONGFORGE_CONTAINER_SOURCE' && bash tests/main_start.sh ${exec_args[*]@Q}" >>"$log_file" 2>&1
status=$?
set -e

result_status=failed
[[ "$status" -eq 0 ]] && result_status=passed
models_json="${models//\\/\\\\}"
models_json="${models_json//\"/\\\"}"
printf '{"status":"%s","environment":"%s","sha":"%s","models":"%s","exit_code":%d,"log":"%s"}\n' \
  "$result_status" "$env_code" "$sha" "$models_json" "$status" "$log_file" >"$result_file"
echo "LOONGFORGE_RESULT_JSON=$result_file"
write_outputs
exit "$status"
