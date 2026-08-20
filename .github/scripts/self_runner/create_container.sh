#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

image="${1:-}"
source_dir="${2:-}"
suite="${3:-}"
container_name="${4:-}"
host_data_root="${LOONGFORGE_HOST_DATA_ROOT:-}"
container_data_root="${LOONGFORGE_CONTAINER_DATA_ROOT:-}"
host_output_root="${LOONGFORGE_HOST_OUTPUT_ROOT:-}"
container_output_root="${LOONGFORGE_CONTAINER_OUTPUT_ROOT:-}"
container_source="${LOONGFORGE_CONTAINER_SOURCE:-}"
container_source_mount="${LOONGFORGE_CONTAINER_SOURCE_MOUNT:-${container_source}-source}"
docker_bin="${DOCKER_BIN:-docker}"
gpu_device="${LOONGFORGE_GPU_DEVICE:-nvidia.com/gpu=all}"

[[ -n "$image" && -d "$source_dir" && "$suite" =~ ^(llm_vlm|embodied)$ && -n "$container_name" ]] || {
  echo "usage: create_container.sh IMAGE SOURCE_DIR SUITE CONTAINER_NAME" >&2
  exit 2
}
[[ -d "$host_data_root" ]] || { echo "LOONGFORGE_HOST_DATA_ROOT is not a directory" >&2; exit 2; }
[[ -d "$host_output_root" ]] || { echo "LOONGFORGE_HOST_OUTPUT_ROOT is not a directory" >&2; exit 2; }
[[ -n "$container_data_root" && -n "$container_output_root" && -n "$container_source" ]] || {
  echo "container data, output, and source paths are required" >&2
  exit 2
}

megatron_dir="$source_dir/third_party/Loong-Megatron"
[[ -f "$megatron_dir/megatron/core/transformer/hyper_connection.py" ]] || {
  echo "Loong-Megatron submodule is not initialized" >&2
  exit 2
}
"$docker_bin" container inspect "$container_name" >/dev/null 2>&1 && {
  echo "container already exists: $container_name" >&2
  exit 2
}
if [[ "${LOONGFORGE_PULL_IMAGE:-true}" == true ]]; then
  "$docker_bin" pull "$image" >/dev/null 2>&1 || {
    echo "failed to pull the configured regression image" >&2
    exit 1
  }
fi

args=(
  run -d
  --name "$container_name"
  --hostname "$container_name"
  --label io.loongforge.ci=true
  --label "io.loongforge.suite=$suite"
  --user root
  --network host
  --cap-add IPC_LOCK
  --ipc host
  --device="$gpu_device"
  -e "LOONGFORGE_TEST_SUITE=$suite"
  -e "TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:?TRITON_LIBCUDA_PATH is required}"
  -v "$host_data_root:$container_data_root:ro"
  -v "$host_output_root:$container_output_root"
  -v "$source_dir:$container_source_mount:ro"
  -v "$megatron_dir:/workspace/Loong-Megatron:ro"
)

# Proxy settings are runner-local and optional. Pass only non-empty values so
# internal-only runners do not alter the container environment.
for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  [[ -n "${!proxy_var:-}" ]] && args+=(--env "$proxy_var=${!proxy_var}")
done

for device in /dev/infiniband/rdma_cm /dev/infiniband/issm0 \
              /dev/infiniband/ucm0 /dev/infiniband/umad0 \
              /dev/infiniband/uverbs0; do
  [[ -e "$device" ]] && args+=(--device "$device:$device")
done

"$docker_bin" "${args[@]}" "$image" sleep infinity >/dev/null 2>&1 || {
  echo "failed to create the regression container" >&2
  exit 1
}
"$docker_bin" exec "$container_name" bash -lc \
  "mkdir -p '$container_source' && \
   find '$container_source' -mindepth 1 -maxdepth 1 ! -name third_party -exec rm -rf -- {} + && \
   mkdir -p '$container_source/third_party' && \
   find '$container_source/third_party' -mindepth 1 -maxdepth 1 ! -name Loong-Megatron -exec rm -rf -- {} + && \
   tar -C '$container_source_mount' --exclude='./third_party/Loong-Megatron' -cf - . | \
     tar -C '$container_source' -xf -"
