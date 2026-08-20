#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=""
suite=""
sha=""
models=""
candidate_revision=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_dir="${2:-}"; shift 2 ;;
    --suite) suite="${2:-}"; shift 2 ;;
    --sha) sha="${2:-}"; shift 2 ;;
    --model) models="${2:-}"; shift 2 ;;
    --candidate-revision) candidate_revision="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$suite" =~ ^(llm_vlm|embodied)$ ]] || { echo "suite must be llm_vlm or embodied" >&2; exit 2; }
[[ -d "$source_dir" && "$sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "source and a full commit SHA are required" >&2
  exit 2
}
[[ "$models" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*(,[A-Za-z0-9][A-Za-z0-9._/-]*)*$ ]] || {
  echo "at least one valid model is required" >&2
  exit 2
}
[[ -z "$candidate_revision" || "$candidate_revision" =~ ^[A-Za-z0-9._:/-]+$ ]] || {
  echo "candidate revision is invalid" >&2
  exit 2
}

: "${LOONGFORGE_DEFAULT_IMAGE:?LOONGFORGE_DEFAULT_IMAGE is required}"
: "${LOONGFORGE_HOST_DATA_ROOT:?LOONGFORGE_HOST_DATA_ROOT is required}"
: "${LOONGFORGE_CONTAINER_DATA_ROOT:?LOONGFORGE_CONTAINER_DATA_ROOT is required}"
: "${LOONGFORGE_HOST_OUTPUT_ROOT:?LOONGFORGE_HOST_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_OUTPUT_ROOT:?LOONGFORGE_CONTAINER_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_SOURCE:?LOONGFORGE_CONTAINER_SOURCE is required}"
: "${LOONGFORGE_RUNNER_LOG_ROOT:?LOONGFORGE_RUNNER_LOG_ROOT is required}"
docker_bin="${DOCKER_BIN:-docker}"
mkdir -p "$LOONGFORGE_RUNNER_LOG_ROOT" "$LOONGFORGE_HOST_OUTPUT_ROOT/resume"

image="${candidate_revision:-$LOONGFORGE_DEFAULT_IMAGE}"
if [[ -n "$candidate_revision" ]]; then
  export LOONGFORGE_PULL_IMAGE=false
fi
container_name="loongforge-ci-${suite}-${sha:0:12}-$$"
log_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.log"
result_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.result.json"
resume_state_file="$LOONGFORGE_CONTAINER_OUTPUT_ROOT/resume/${container_name}.json"

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
  if [[ -n "$candidate_revision" ]]; then
    "$docker_bin" image rm "$candidate_revision" >/dev/null 2>&1 || true
  fi
  if [[ ! -f "$result_file" ]]; then
    printf '{"status":"failed","suite":"%s","sha":"%s","models":"%s","exit_code":%d,"log":"%s"}\n' \
      "$suite" "$sha" "$models" "$status" "${container_name}.log" >"$result_file"
  fi
  "$script_dir/cleanup.sh" >>"$log_file" 2>&1 || true
  write_outputs
  exit "$status"
}
trap cleanup EXIT INT TERM

"$script_dir/create_container.sh" "$image" "$source_dir" "$suite" "$container_name" >>"$log_file" 2>&1

set +e
if [[ "$suite" == embodied ]]; then
  [[ -x "$source_dir/tests/embodied/run.sh" ]] || {
    echo "embodied test suite is missing: tests/embodied/run.sh" >&2
    exit 2
  }
  read -r -a model_args <<<"${models//,/ }"
  test_entry="tests/embodied/run.sh"
  exec_args=(--chip "${LOONGFORGE_BASELINE_EMBODIED:-P6K}" --models "${model_args[@]}")
  extra_env=(
    -e "LOCAL_VLA_ARTIFACTS_ROOT=$LOONGFORGE_CONTAINER_DATA_ROOT"
    -e "EMBODIED_LOG_ROOT=$LOONGFORGE_CONTAINER_OUTPUT_ROOT/embodied"
  )
else
  # main.py resolves configs/, tasks/, and optional_configs/ relative to the
  # llm_vlm suite directory, so run it from there rather than repo root.
  [[ -f "$source_dir/tests/llm_vlm/main.py" ]] || {
    echo "LLM/VLM test suite is missing: tests/llm_vlm/main.py" >&2
    exit 2
  }
  test_entry="cd tests/llm_vlm && python3 main.py"
  read -r -a model_args <<<"${models//,/ }"
  exec_args=(--models "${model_args[@]}" --chip "${LOONGFORGE_BASELINE_LLM_VLM:-A800}" \
    --tasks check_correctness_task check_precess_data_task --training_type pretrain sft \
    --node_nums 1 --gpu_nums 8 --check_loss_only)
  extra_env=()
fi

"$docker_bin" exec "${extra_env[@]}" \
  -e "PFS_PATH=$LOONGFORGE_CONTAINER_DATA_ROOT" \
  -e "TRAINING_LOG_PATH=$LOONGFORGE_CONTAINER_OUTPUT_ROOT" \
  -e "LOONGFORGE_TEST_SUITE=$suite" \
  -e "RESUME_STATE_FILE=$resume_state_file" \
  "$container_name" \
  bash -lc "cd '$LOONGFORGE_CONTAINER_SOURCE' && $test_entry ${exec_args[*]@Q}" >>"$log_file" 2>&1
status=$?
set -e

result_status=failed
[[ "$status" -eq 0 ]] && result_status=passed
models_json="${models//\\/\\\\}"
models_json="${models_json//\"/\\\"}"
printf '{"status":"%s","suite":"%s","sha":"%s","models":"%s","exit_code":%d,"log":"%s"}\n' \
  "$result_status" "$suite" "$sha" "$models_json" "$status" "${container_name}.log" >"$result_file"
write_outputs
exit "$status"
