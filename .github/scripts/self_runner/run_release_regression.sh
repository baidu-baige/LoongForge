#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
execution_slot="${LOONGFORGE_RELEASE_EXECUTION_SLOT:-manual}"
image=""
source_dir=""
sha=""
llm_model="deepseek_v2_lite"
vla_model="pi05_ddp"
request_baseline_chip=""
master_port=$((20000 + ($$ % 30000)))

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="${2:-}"; shift 2 ;;
    --source) source_dir="${2:-}"; shift 2 ;;
    --sha) sha="${2:-}"; shift 2 ;;
    --llm-model) llm_model="${2:-}"; shift 2 ;;
    --vla-model) vla_model="${2:-}"; shift 2 ;;
    --baseline-chip) request_baseline_chip="${2:-}"; shift 2 ;;
    *) printf '%s\n' 'argument: invalid' >&2; exit 2 ;;
  esac
done

[[ "$image" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] || {
  printf '%s\n' 'image: invalid' >&2
  exit 2
}
[[ -d "$source_dir" && "$sha" =~ ^[0-9a-f]{40}$ ]] || {
  printf '%s\n' 'source and a full commit SHA are required' >&2
  exit 2
}
[[ "$llm_model" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || {
  printf '%s\n' 'llm model: invalid' >&2
  exit 2
}
[[ "$vla_model" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || {
  printf '%s\n' 'vla model: invalid' >&2
  exit 2
}
[[ "$request_baseline_chip" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  printf '%s\n' 'baseline chip: required and must be a directory identifier' >&2
  exit 2
}

run_id="${GITHUB_RUN_ID:-manual-$(date -u +%Y%m%d%H%M%S)-$$}"
run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ && "$run_attempt" =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' 'run identity: invalid' >&2
  exit 2
}
run_key="$run_id-attempt-$run_attempt"

mode="${LOONGFORGE_EXECUTION_MODE:-}"
if [[ "$mode" == ssh ]]; then
  alias="${LOONGFORGE_EXECUTION_ALIAS:?execution slot is incomplete}"
  root="${LOONGFORGE_EXECUTION_ROOT:?execution slot is incomplete}"
  remote_source="${LOONGFORGE_EXECUTION_SOURCE:?execution slot is incomplete}"
  remote_config="${LOONGFORGE_EXECUTION_CONFIG:?execution slot is incomplete}"
  [[ "$alias" =~ ^[A-Za-z0-9._@-]+$ ]] || {
    printf '%s\n' 'execution-slot: invalid configuration' >&2
    exit 2
  }
  for path_name in root remote_source remote_config; do
    [[ "${!path_name}" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
      printf '%s\n' 'execution-slot: invalid configuration' >&2
      exit 2
    }
  done
  work_dir="$root/runs/$run_id"
  artifact_dir="$PWD/loongforge-release-artifacts"

  if [[ -L "$artifact_dir" || (-e "$artifact_dir" && ! -d "$artifact_dir") ]]; then
    printf '%s\n' 'artifact: invalid directory' >&2
    exit 2
  fi
  rm -rf "$artifact_dir"
  mkdir -p "$artifact_dir"

  ssh_options=(-o BatchMode=yes -o ConnectTimeout=15)
  ssh "${ssh_options[@]}" "$alias" "mkdir -p '$work_dir' '$remote_source'"
  rsync --archive --delete --delete-excluded \
    --exclude='.git/' --exclude='.pytest_cache/' --exclude='__pycache__/' \
    "$source_dir/" "$alias:$remote_source/"

  status=0
  ssh "${ssh_options[@]}" "$alias" \
    "cd '$work_dir' && env -u DOCKER_HOST -u LOONGFORGE_EXECUTION_MODE \
     LOONGFORGE_RELEASE_EXECUTION_SLOT='$execution_slot' \
     GITHUB_RUN_ID='$run_id' GITHUB_RUN_ATTEMPT='$run_attempt' \
     CI_CONFIG_PATH='$remote_config' \
     '$remote_source/.github/scripts/self_runner/run_release_regression.sh' \
     --image '$image' --source '$remote_source' --sha '$sha' \
     --llm-model '$llm_model' --vla-model '$vla_model' \
     --baseline-chip '$request_baseline_chip'" || status=$?

  if ! rsync --archive "$alias:$work_dir/loongforge-release-artifacts/" "$artifact_dir/"; then
    status=1
  fi
  evidence_root="${LOONGFORGE_RELEASE_EVIDENCE_ROOT:-}"
  if [[ -n "$evidence_root" && "$evidence_root" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    mkdir -p "$evidence_root/$run_id/loongforge-release-artifacts"
    if ! rsync --archive --delete \
      "$artifact_dir/" "$evidence_root/$run_id/loongforge-release-artifacts/"; then
      status=1
    fi
  fi
  exit "$status"
fi

[[ -z "$mode" || "$mode" == local ]] || {
  printf '%s\n' 'execution-slot: invalid' >&2
  exit 2
}

# shellcheck disable=SC1091
source "$script_dir/../load_ci_config.sh"

: "${LOONGFORGE_DEFAULT_IMAGE:?LOONGFORGE_DEFAULT_IMAGE is required}"
: "${LOONGFORGE_HOST_DATA_ROOT:?LOONGFORGE_HOST_DATA_ROOT is required}"
: "${LOONGFORGE_CONTAINER_DATA_ROOT:?LOONGFORGE_CONTAINER_DATA_ROOT is required}"
: "${LOONGFORGE_HOST_LLM_DATA_ROOT:?LOONGFORGE_HOST_LLM_DATA_ROOT is required}"
: "${LOONGFORGE_CONTAINER_LLM_DATA_ROOT:?LOONGFORGE_CONTAINER_LLM_DATA_ROOT is required}"
: "${LOONGFORGE_HOST_OUTPUT_ROOT:?LOONGFORGE_HOST_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_OUTPUT_ROOT:?LOONGFORGE_CONTAINER_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_SOURCE:?LOONGFORGE_CONTAINER_SOURCE is required}"
: "${LOONGFORGE_RUNNER_LOG_ROOT:?LOONGFORGE_RUNNER_LOG_ROOT is required}"

docker_bin="${DOCKER_BIN:-docker}"
llm_output="$LOONGFORGE_CONTAINER_OUTPUT_ROOT/llm_vlm/run-$run_key"
vla_output="$LOONGFORGE_CONTAINER_OUTPUT_ROOT/embodied/run-$run_key"
artifact_dir="$PWD/loongforge-release-artifacts"
if [[ -L "$artifact_dir" || (-e "$artifact_dir" && ! -d "$artifact_dir") ]]; then
  printf '%s\n' 'artifact: invalid directory' >&2
  exit 2
fi
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir" "$LOONGFORGE_RUNNER_LOG_ROOT" "$LOONGFORGE_HOST_OUTPUT_ROOT"
mkdir -p "$LOONGFORGE_HOST_OUTPUT_ROOT/llm_vlm/run-$run_key" \
  "$LOONGFORGE_HOST_OUTPUT_ROOT/embodied/run-$run_key"

"$script_dir/preflight.sh" release false >/dev/null

container_name="loongforge-release-${sha:0:12}-$$"
llm_log="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.llm.log"
vla_log="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.vla.log"
result_raw="$artifact_dir/${container_name}.result.json"
image_digest="$("$docker_bin" image inspect \
  --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' "$image")"
status=1

redact_artifact() {
  local input="$1"
  local output="$2"
  if [[ -f "$input" ]]; then
    python3 "$script_dir/../../../ci/redact_ci_artifact.py" \
      --input "$input" --output "$output" || true
  fi
}

cleanup() {
  local exit_status="$status"
  set +e
  if ! python3 "$script_dir/../../../ci/release_evidence.py" \
    --source "$source_dir" --llm-log "$llm_log" --llm-model "$llm_model" \
    --llm-chip "$request_baseline_chip" \
    --vla-results "$LOONGFORGE_HOST_OUTPUT_ROOT/embodied/run-$run_key/results.json" \
    --vla-chip "$request_baseline_chip" --exit-code "${llm_status:-$exit_status}" \
    --output "$artifact_dir/.regression-metrics.json" >/dev/null 2>&1; then
    printf '%s\n' 'release-evidence: metric collection failed' >&2
  fi
  if [[ -f "$artifact_dir/.regression-metrics.json" ]]; then
    redact_artifact "$artifact_dir/.regression-metrics.json" "$artifact_dir/regression-metrics.json"
    unlink "$artifact_dir/.regression-metrics.json"
  fi
  "$docker_bin" rm -f "$container_name" >/dev/null 2>&1 || true
  redact_artifact "$llm_log" "$artifact_dir/llm-regression.log"
  redact_artifact "$vla_log" "$artifact_dir/vla-regression.log"
  if [[ ! -f "$result_raw" ]]; then
    printf \
      '{"status":"failed","suite":"release","models":"%s,%s","source_sha":"%s","execution_slot":"%s","image_digest":"%s","exit_code":%d}\n' \
      "$llm_model" "$vla_model" "$sha" "$execution_slot" "$image_digest" \
      "$exit_status" >"$result_raw"
  fi
  redact_artifact "$result_raw" "$artifact_dir/release-regression.result.json"
  [[ ! -f "$result_raw" ]] || unlink "$result_raw"
  if [[ -n "${LOONGFORGE_RELEASE_EVIDENCE_ROOT:-}" ]]; then
    local archive="$LOONGFORGE_RELEASE_EVIDENCE_ROOT/$run_id/loongforge-release-artifacts"
    mkdir -p "$archive"
    if [[ ! -f "$artifact_dir/regression-metrics.json" && -f "$archive/regression-metrics.json" ]]; then
      unlink "$archive/regression-metrics.json"
    fi
    rsync --archive "$artifact_dir/" \
      "$archive/" >/dev/null 2>&1
  fi
  "$script_dir/cleanup.sh" >/dev/null 2>&1 || true
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s\n' 'artifact_dir=loongforge-release-artifacts' >>"$GITHUB_OUTPUT"
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

export LOONGFORGE_PULL_IMAGE=false
"$script_dir/create_container.sh" "$image" "$source_dir" release "$container_name" \
  >>"$llm_log" 2>&1

llm_model_q=$(printf '%q' "$llm_model")
llm_chip_q=$(printf '%q' "$request_baseline_chip")
container_source="${LOONGFORGE_CONTAINER_SOURCE%/}"
container_source_q=$(printf '%q' "$container_source")
output_root_q=$(printf '%q' "$LOONGFORGE_CONTAINER_OUTPUT_ROOT")

set +e
"$docker_bin" exec \
  -e "PFS_PATH=$LOONGFORGE_CONTAINER_LLM_DATA_ROOT" \
  -e "TRAINING_LOG_PATH=$llm_output" \
  -e "LOONGFORGE_PATH=$container_source/" \
  -e "MEGATRON_PATH=$container_source/third_party/Loong-Megatron" \
  -e "RANK=0" \
  -e "WORLD_SIZE=1" \
  -e "MASTER_ADDR=127.0.0.1" \
  -e "MASTER_PORT=$master_port" \
  -e "LOONGFORGE_TEST_SUITE=llm_vlm" \
  "$container_name" \
  bash -lc \
  "cd ${container_source_q} && cd tests/llm_vlm && \
   python3 main.py --models ${llm_model_q} --chip ${llm_chip_q} \
   --tasks check_correctness_task check_precess_data_task \
   --training_type pretrain sft --node_nums 1 --gpu_nums 8 --check_loss_only" \
  >"$llm_log" 2>&1
llm_status=$?
set -e

vla_status=1
if [[ "$llm_status" -eq 0 ]]; then
  vla_model_q=$(printf '%q' "$vla_model")
  vla_chip_q=$(printf '%q' "$request_baseline_chip")
  vla_data_q=$(printf '%q' "$LOONGFORGE_CONTAINER_DATA_ROOT")
  set +e
  "$docker_bin" exec \
    -e "LOCAL_VLA_ARTIFACTS_ROOT=$LOONGFORGE_CONTAINER_DATA_ROOT" \
    -e "EMBODIED_LOG_ROOT=$vla_output" \
    -e "LOONGFORGE_TEST_SUITE=embodied" \
    "$container_name" \
    bash -lc \
    "cd ${container_source_q} && \
     LOCAL_VLA_ARTIFACTS_ROOT=${vla_data_q} \
     EMBODIED_LOG_ROOT=${output_root_q}/embodied \
     bash tests/embodied/run.sh --chip ${vla_chip_q} --models ${vla_model_q} \
     --log_dir $(printf '%q' "$vla_output")" \
    >"$vla_log" 2>&1
  vla_status=$?
  set -e
fi

if [[ "$llm_status" -ne 0 ]]; then
  status="$llm_status"
elif [[ "$vla_status" -ne 0 ]]; then
  status="$vla_status"
else
  status=0
fi

printf '{"status":"%s","suite":"release","models":"%s,%s","source_sha":"%s","execution_slot":"%s","image_digest":"%s","exit_code":%d}\n' \
  "$([[ "$status" -eq 0 ]] && printf passed || printf failed)" \
  "$llm_model" "$vla_model" "$sha" "$execution_slot" "$image_digest" \
  "$status" >"$result_raw"
exit "$status"
