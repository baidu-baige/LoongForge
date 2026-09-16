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
    *) printf '%s\n' 'argument: invalid' >&2; exit 2 ;;
  esac
done

[[ "$suite" =~ ^(mcore|torch)$ ]] || { echo "suite must be mcore or torch" >&2; exit 2; }
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
artifact_dir="$PWD/loongforge-artifacts"
if [[ -L "$artifact_dir" || (-e "$artifact_dir" && ! -d "$artifact_dir") ]]; then
  printf '%s\n' 'artifact: invalid directory' >&2
  exit 2
fi
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir"

# Validate the runner and default image before creating any container. A
# candidate image was already preflighted by the workflow before it was built.
"$script_dir/preflight.sh" "$suite" false >/dev/null

image="${candidate_revision:-$LOONGFORGE_DEFAULT_IMAGE}"
if [[ -n "$candidate_revision" ]]; then
  export LOONGFORGE_PULL_IMAGE=false
fi
container_name="loongforge-ci-${suite}-${sha:0:12}-$$"
log_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.log"
result_file="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.result.json"
resume_state_file="$LOONGFORGE_CONTAINER_OUTPUT_ROOT/resume/${container_name}.json"
artifact_result="$artifact_dir/${container_name}.result.json"
artifact_log="$artifact_dir/${container_name}.log"

status=1
outputs_written=false
write_outputs() {
  if [[ "$outputs_written" == false && -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s\n' 'artifact_dir=loongforge-artifacts' >>"$GITHUB_OUTPUT"
    outputs_written=true
  fi
}
cleanup() {
  mkdir -p "$artifact_dir"
  if [[ ! -f "$log_file" ]]; then
    docker_log_tmp="$artifact_dir/.docker-logs.$$"
    if "$docker_bin" logs "$container_name" >"$docker_log_tmp" 2>&1; then
      mv "$docker_log_tmp" "$log_file"
    else
      rm -f "$docker_log_tmp"
    fi
  fi
  "$docker_bin" rm -f "$container_name" >/dev/null 2>&1 || true
  if [[ -n "$candidate_revision" ]]; then
    "$docker_bin" image rm "$candidate_revision" >/dev/null 2>&1 || true
  fi
  if [[ ! -f "$result_file" ]]; then
    printf '%s\n' 'artifact: missing raw result' >&2
    status=1
  elif [[ ! -f "$log_file" ]]; then
    printf '%s\n' 'artifact: missing raw log' >&2
    status=1
  else
    python3 "$script_dir/../../../ci/redact_ci_artifact.py" --input "$result_file" --output "$artifact_result" || status=1
    python3 "$script_dir/../../../ci/redact_ci_artifact.py" --input "$log_file" --output "$artifact_log" || status=1
  fi
  "$script_dir/cleanup.sh" >/dev/null 2>&1 || true
  write_outputs
  exit "$status"
}
trap cleanup EXIT INT TERM

"$script_dir/create_container.sh" "$image" "$source_dir" "$suite" "$container_name" >>"$log_file" 2>&1

set +e
if [[ "$suite" == torch ]]; then
  # The Torch entry script is not executable in git (mode 100644), so
  # invoke it through bash instead of relying on the exec bit.
  [[ -f "$source_dir/tests/torch/run.sh" ]] || {
    echo "Torch test suite is missing: tests/torch/run.sh" >&2
    exit 2
  }
  read -r -a model_args <<<"${models//,/ }"
  test_entry="bash tests/torch/run.sh"
  exec_args=(--chip "${LOONGFORGE_BASELINE_TORCH:-p}" --models "${model_args[@]}")
  extra_env=(
    -e "LOCAL_VLA_ARTIFACTS_ROOT=$LOONGFORGE_CONTAINER_DATA_ROOT"
    -e "TORCH_LOG_ROOT=$LOONGFORGE_CONTAINER_OUTPUT_ROOT/torch"
  )
else
  # main.py resolves configs/, tasks/, and optional_configs/ relative to the
  # mcore suite directory, so run it from there rather than repo root.
  [[ -f "$source_dir/tests/mcore/main.py" ]] || {
    echo "MCore test suite is missing: tests/mcore/main.py" >&2
    exit 2
  }
  test_entry="cd tests/mcore && python3 main.py"
  read -r -a model_args <<<"${models//,/ }"
  exec_args=(--models "${model_args[@]}" --chip "${LOONGFORGE_BASELINE_MCORE:-a}" \
    --tasks check_correctness_task check_precess_data_task --training_type pretrain sft \
    --node_nums 1 --gpu_nums 8 --check_loss_only)
  extra_env=()
fi

exec_command=( "$docker_bin" exec )
if (( ${#extra_env[@]} )); then
  exec_command+=( "${extra_env[@]}" )
fi
exec_command+=( \
  -e "PFS_PATH=$LOONGFORGE_CONTAINER_DATA_ROOT" \
  -e "TRAINING_LOG_PATH=$LOONGFORGE_CONTAINER_OUTPUT_ROOT" \
  -e "LOONGFORGE_TEST_SUITE=$suite" \
  -e "RESUME_STATE_FILE=$resume_state_file" \
  "$container_name" \
  bash -lc "cd '$LOONGFORGE_CONTAINER_SOURCE' && $test_entry ${exec_args[*]@Q}" )
"${exec_command[@]}" >>"$log_file" 2>&1
status=$?
set -e

suite_results_copied=false
if [[ "$suite" == torch ]]; then
  # Surface the regression framework's per-model results (loss/grad_norm
  # baseline comparisons) so the workflow can report them on the pull
  # request check run. The job concurrency group keeps parallel torch
  # runs off this runner, so the newest run directory is this container's.
  newest_results="$(ls -1t "$LOONGFORGE_HOST_OUTPUT_ROOT"/torch/run_*/results.json 2>/dev/null | head -1 || true)"
  if [[ -n "$newest_results" && -f "$newest_results" ]]; then
    if python3 "$script_dir/../../../ci/redact_ci_artifact.py" \
        --input "$newest_results" --output "$artifact_dir/suite-results.json"; then
      suite_results_copied=true
    fi
  fi
  # Preserve the tail of each model's training log so a failed run can be
  # diagnosed from the artifact alone (OOM tracebacks and elastic-agent
  # failure summaries only appear in train.log, not in the suite log).
  if [[ -n "$newest_results" && -d "$(dirname "$newest_results")" ]]; then
    for train_log in "$(dirname "$newest_results")"/*/train.log; do
      [[ -f "$train_log" ]] || continue
      model_name="$(basename "$(dirname "$train_log")")"
      train_tail_tmp="$artifact_dir/.train-${model_name}.$$"
      if tail -c 65536 "$train_log" >"$train_tail_tmp" 2>/dev/null; then
        python3 "$script_dir/../../../ci/redact_ci_artifact.py" \
          --input "$train_tail_tmp" \
          --output "$artifact_dir/train-${model_name}.log.tail" || true
      fi
      rm -f "$train_tail_tmp"
    done
  fi
fi

result_status=failed
[[ "$status" -eq 0 ]] && result_status=passed
models_json="${models//\\/\\\\}"
models_json="${models_json//\"/\\\"}"
printf '{"status":"%s","suite":"%s","sha":"%s","models":"%s","exit_code":%d,"log":"%s"}\n' \
  "$result_status" "$suite" "$sha" "$models_json" "$status" "${container_name}.log" >"$result_file"
if [[ "$suite_results_copied" == true && -n "${GITHUB_OUTPUT:-}" ]]; then
  printf '%s\n' 'suite_results=loongforge-artifacts/suite-results.json' >>"$GITHUB_OUTPUT"
fi
write_outputs
exit "$status"
