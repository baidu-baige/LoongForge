#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target=""
head_sha=""
tree_sha=""
pr_number=""
source_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --sha) head_sha="$2"; shift 2 ;;
    --tree-sha) tree_sha="$2"; shift 2 ;;
    --pr) pr_number="$2"; shift 2 ;;
    --source) source_dir="$2"; shift 2 ;;
    *) echo "unknown builder argument: $1" >&2; exit 2 ;;
  esac
done

case "$target" in
  a|ampere) compile_env=ampere; target_code=a ;;
  h|hopper) compile_env=hopper; target_code=h ;;
  p|blackwell) compile_env=blackwell; target_code=p ;;
  *) echo "unsupported image target: $target" >&2; exit 2 ;;
esac

[[ -n "$head_sha" && -n "$tree_sha" && -n "$pr_number" && -d "$source_dir" ]] || {
  echo "builder requires --pr, --sha, --tree-sha and --source" >&2
  exit 2
}

config="${CI_CONFIG_PATH_IMAGE:-${CI_CONFIG_PATH:-}}"
if [[ -n "$config" ]]; then
  [[ -f "$config" ]] || { echo "image config not found: $config" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$config"
  set +a
fi

repo_var="IMAGE_REPOSITORY_${target_code^^}"
repository="${!repo_var:-${CI_CANDIDATE_IMAGE_REPOSITORY:-}}"
registry="${IMAGE_REGISTRY:-}"
[[ -n "$repository" ]] || { echo "$repo_var is required" >&2; exit 2; }
if [[ -n "${INTERNAL_REGISTRY_USERNAME:-}" || -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]]; then
  [[ -n "$registry" && -n "${INTERNAL_REGISTRY_USERNAME:-}" && -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]] || {
    echo "registry host, username and password must be configured together" >&2
    exit 2
  }
  printf '%s' "$INTERNAL_REGISTRY_PASSWORD" | docker login "$registry" \
    --username "$INTERNAL_REGISTRY_USERNAME" --password-stdin >/dev/null
fi

short_head="${head_sha:0:12}"
short_tree="${tree_sha:0:12}"
tag="${CANDIDATE_TAG_PREFIX:-pr}-${pr_number}-head-${short_head}-${target_code}-${short_tree}"
if [[ -n "$registry" && "$repository" != "$registry/"* ]]; then
  image_repository="$registry/$repository"
else
  image_repository="$repository"
fi
image_ref="$image_repository:$tag"

source_dockerfile="${IMAGE_DOCKERFILE:-$source_dir/docker/Dockerfile}"
[[ -f "$source_dockerfile" ]] || { echo "Dockerfile not found: $source_dockerfile" >&2; exit 2; }
context_dir="$(mktemp -d)"
mkdir -p "$context_dir/LoongForge"
tar -C "$source_dir" \
  --exclude=.git \
  --exclude=.pytest_cache \
  --exclude='*.log' \
  --exclude='__pycache__' \
  -cf - . | tar -C "$context_dir/LoongForge" -xf -
trap 'rm -rf "$context_dir"' EXIT
if [[ -n "${IMAGE_DOCKERFILE:-}" ]]; then
  dockerfile="$source_dockerfile"
else
  dockerfile="$context_dir/LoongForge/docker/Dockerfile"
fi

build_args=(
  --build-arg "COMPILE_ENV=$compile_env"
  --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  --label "org.opencontainers.image.revision=$head_sha"
  --label "io.loongforge.pr-number=$pr_number"
  --label "io.loongforge.tree-sha=$tree_sha"
  --label "io.loongforge.image-target=$target_code"
  --label "io.loongforge.image-revision=$tag"
)
[[ -n "${IMAGE_BASE_IMAGE:-}" ]] && build_args+=(--build-arg "BASE_IMAGE=$IMAGE_BASE_IMAGE")
for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  [[ -n "${!proxy_var:-}" ]] && build_args+=(--build-arg "$proxy_var=${!proxy_var}")
done
for build_arg_var in $(compgen -A variable IMAGE_BUILD_ARG_ | sort); do
  arg_name="${build_arg_var#IMAGE_BUILD_ARG_}"
  [[ -n "$arg_name" && -n "${!build_arg_var:-}" ]] || continue
  build_args+=(--build-arg "$arg_name=${!build_arg_var}")
done
for secret_spec in \
  "IMAGE_APT_SOURCES:apt_sources" \
  "IMAGE_PIP_CONFIG:pip_config" \
  "IMAGE_SOURCE_MANIFEST:source_manifest"; do
  config_var="${secret_spec%%:*}"
  secret_id="${secret_spec#*:}"
  secret_path="${!config_var:-}"
  [[ -n "$secret_path" ]] || continue
  [[ -f "$secret_path" ]] || { echo "$config_var file not found: $secret_path" >&2; exit 2; }
  build_args+=(--secret "id=$secret_id,src=$secret_path")
done

if [[ "${IMAGE_DRY_RUN:-false}" == true ]]; then
  printf 'docker build -f %q -t %q %q' "$dockerfile" "$image_ref" "$context_dir" >&2
  printf ' %q' "${build_args[@]}" >&2
  printf '\n' >&2
else
  DOCKER_BUILDKIT=1 docker build "${build_args[@]}" -f "$dockerfile" -t "$image_ref" "$context_dir" >&2
  if [[ "${IMAGE_PUSH:-true}" == true ]]; then
    docker push "$image_ref" >&2
  fi
fi

printf '%s\n' "$image_ref"
