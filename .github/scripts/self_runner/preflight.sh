#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

suite="${1:-}"
build_image="${2:-false}"
[[ "$suite" =~ ^(llm_vlm|native)$ ]] || { printf '%s\n' 'suite: invalid' >&2; exit 2; }
[[ "$build_image" == true || "$build_image" == false ]] || {
  printf '%s\n' 'build_image: invalid' >&2
  exit 2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$script_dir/../load_ci_config.sh"

required=(
  LOONGFORGE_DEFAULT_IMAGE
  LOONGFORGE_HOST_DATA_ROOT
  LOONGFORGE_CONTAINER_DATA_ROOT
  LOONGFORGE_HOST_OUTPUT_ROOT
  LOONGFORGE_CONTAINER_OUTPUT_ROOT
  LOONGFORGE_CONTAINER_SOURCE
  LOONGFORGE_RUNNER_LOG_ROOT
  TRITON_LIBCUDA_PATH
  LOONGFORGE_GPU_DEVICE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf '%s\n' "$name" >&2
    exit 2
  fi
done

for name in LOONGFORGE_HOST_DATA_ROOT LOONGFORGE_HOST_OUTPUT_ROOT LOONGFORGE_RUNNER_LOG_ROOT TRITON_LIBCUDA_PATH; do
  if [[ ! -e "${!name}" ]]; then
    printf '%s\n' 'mounts: missing' >&2
    exit 2
  fi
done

docker_bin="${DOCKER_BIN:-docker}"
if ! "$docker_bin" version >/dev/null 2>&1; then
  printf '%s\n' 'docker: missing' >&2
  exit 1
fi
printf '%s\n' 'docker: ok'

if [[ "$build_image" == true ]]; then
  if [[ "${LOONGFORGE_ALLOW_PR_IMAGE_BUILD:-false}" != true ]]; then
    printf '%s\n' 'build_image: disabled' >&2
    exit 2
  fi
  if ! "$docker_bin" buildx version >/dev/null 2>&1; then
    printf '%s\n' 'buildx: missing' >&2
    exit 1
  fi
  printf '%s\n' 'buildx: ok'

  if [[ "${LOONGFORGE_MIN_DOCKER_FREE_GB+x}" ]]; then
    min_docker_free_gb="$LOONGFORGE_MIN_DOCKER_FREE_GB"
  else
    min_docker_free_gb=250
  fi
  "$script_dir/../check_docker_storage.sh" "$min_docker_free_gb"
fi

if ! "$docker_bin" run --rm --device="$LOONGFORGE_GPU_DEVICE" "$LOONGFORGE_DEFAULT_IMAGE" true >/dev/null 2>&1; then
  printf '%s\n' 'image-device: invalid' >&2
  exit 1
fi
printf '%s\n' 'image-device: ok'

# Shared runners host other tenants' containers; fail fast with an
# actionable message when any GPU has less free memory than the suite
# needs, instead of burning a full training run on a late OOM. Set
# LOONGFORGE_MIN_FREE_GPU_MB (MiB per GPU) in the runner config to
# override; an empty value disables the check.
if [[ "${LOONGFORGE_MIN_FREE_GPU_MB+x}" ]]; then
  # An explicitly empty value is the documented way to disable this optional
  # host-occupancy check; only an unset variable receives the default.
  min_free_mb="$LOONGFORGE_MIN_FREE_GPU_MB"
else
  min_free_mb=60000
fi
if [[ -n "${LOONGFORGE_MIN_FREE_GPU_MB:-}" && ! "$min_free_mb" =~ ^[0-9]{1,9}$ ]]; then
  printf '%s\n' 'gpu-memory: invalid configuration' >&2
  exit 2
fi
if [[ "$min_free_mb" =~ ^[0-9]{1,9}$ ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'gpu-memory: unavailable' >&2
    exit 1
  fi
  if ! free_line=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null); then
    printf '%s\n' 'gpu-memory: unavailable' >&2
    exit 1
  elif [[ -z "${free_line//$'\n'/}" ]]; then
    # nvidia-smi exists but reports no GPUs at all; a GPU suite runner in
    # that state cannot train. Fail closed instead of passing vacuously.
    printf '%s\n' 'gpu-memory: unavailable' >&2
    exit 1
  else
    insufficient=false
    while IFS= read -r free_mb; do
      # nvidia-smi reports "[N/A]" or "[Not Supported]" for GPUs in an
      # error or uninitialized state; treat those as unavailable rather
      # than letting arithmetic coercion pass the check (fail closed).
      if [[ ! "$free_mb" =~ ^[0-9]{1,9}$ ]]; then
        insufficient=true
      elif (( free_mb < min_free_mb )); then
        insufficient=true
      fi
    done <<<"$free_line"
    if [[ "$insufficient" == true ]]; then
      printf '%s\n' 'gpu-memory: insufficient' >&2
      exit 1
    fi
  fi
  printf '%s\n' 'gpu-memory: ok'
fi
printf '%s\n' 'mounts: ok'
