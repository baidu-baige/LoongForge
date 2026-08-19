#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

image="${1:-}"
source_dir="${2:-}"
env_code="${3:-}"
container_name="${4:-}"
host_pfs="${LOONGFORGE_PFS_ROOT:-}"
container_pfs="${LOONGFORGE_CONTAINER_PFS:-}"
container_source="${LOONGFORGE_CONTAINER_SOURCE:-}"
container_source_mount="${LOONGFORGE_CONTAINER_SOURCE_MOUNT:-${container_source}-source}"
docker_bin="${DOCKER_BIN:-docker}"

[[ -n "$image" && -d "$source_dir" && -n "$env_code" && -n "$container_name" ]] || {
  echo "usage: create_container.sh IMAGE SOURCE_DIR ENV CONTAINER_NAME" >&2
  exit 2
}
[[ -d "$host_pfs" ]] || { echo "LOONGFORGE_PFS_ROOT is not a directory" >&2; exit 2; }
[[ -n "$container_pfs" && -n "$container_source" ]] || {
  echo "container mount variables are required" >&2
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
  if ! "$docker_bin" pull "$image" >&2; then
    if [[ -n "${LOONGFORGE_BOOTSTRAP_IMAGE:-}" && "$image" == "${LOONGFORGE_DEFAULT_IMAGE:-}" ]]; then
      echo "default image is unavailable; using bootstrap image" >&2
      image="$LOONGFORGE_BOOTSTRAP_IMAGE"
      "$docker_bin" pull "$image" >&2
    else
      exit 1
    fi
  fi
fi

args=(
  run -d
  --name "$container_name"
  --hostname "$container_name"
  --label io.loongforge.ci=true
  --label "io.loongforge.environment=$env_code"
  --privileged
  --user root
  --network host
  --shm-size=32768m
  --cap-add IPC_LOCK
  --ipc host
  --gpus all
  -e "LOONGFORGE_TEST_ENV=$env_code"
  -e "TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:?TRITON_LIBCUDA_PATH is required}"
  -v "$host_pfs:$container_pfs"
  -v "$source_dir:$container_source_mount:ro"
  -v "$megatron_dir:/workspace/Loong-Megatron:ro"
  -v /dev/shm:/dev/shm
)

for device in /dev/infiniband/rdma_cm /dev/infiniband/issm0 \
              /dev/infiniband/ucm0 /dev/infiniband/umad0 \
              /dev/infiniband/uverbs0; do
  [[ -e "$device" ]] && args+=(--device "$device:$device")
done

"$docker_bin" "${args[@]}" "$image" sleep infinity
"$docker_bin" exec "$container_name" bash -lc \
  "find '$container_source' -mindepth 1 -maxdepth 1 ! -name third_party -exec rm -rf -- {} + && \
   mkdir -p '$container_source/third_party' && \
   find '$container_source/third_party' -mindepth 1 -maxdepth 1 ! -name Loong-Megatron -exec rm -rf -- {} + && \
   tar -C '$container_source_mount' --exclude='./third_party/Loong-Megatron' -cf - . | \
     tar -C '$container_source' -xf -"
