#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target=""
head_sha=""
tree_sha=""
pr_number=""
source_dir=""
image_ref_override=""
release_mode=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --sha) head_sha="$2"; shift 2 ;;
    --tree-sha) tree_sha="$2"; shift 2 ;;
    --pr) pr_number="$2"; shift 2 ;;
    --source) source_dir="$2"; shift 2 ;;
    --image-ref) image_ref_override="$2"; shift 2 ;;
    --release) release_mode=true; shift ;;
    *) printf '%s\n' 'argument: invalid' >&2; exit 2 ;;
  esac
done

case "$target" in
  a|p|auto) ;;
  *) printf '%s\n' 'target: invalid' >&2; exit 2 ;;
esac

[[ -n "$head_sha" && -n "$tree_sha" && -d "$source_dir" ]] || {
  echo "builder requires --sha, --tree-sha and --source" >&2
  exit 2
}
if [[ "$release_mode" == false && -z "$pr_number" ]]; then
  echo "builder requires --pr for candidate images" >&2
  exit 2
fi

# shellcheck disable=SC1091
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../load_ci_config.sh"

if [[ "$release_mode" == true ]]; then
  [[ "${LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD:-false}" == true ]] || {
    echo "release image builds are disabled on this runner" >&2
    exit 2
  }
else
  [[ "${LOONGFORGE_ALLOW_PR_IMAGE_BUILD:-false}" == true ]] || {
    echo "candidate image builds are disabled on this runner" >&2
    exit 2
  }
fi

requested_target="$target"
detected_target="$($script_dir/detect_gpu_target.sh)" || exit 1
if [[ "$requested_target" == auto ]]; then
  target="$detected_target"
elif [[ "$requested_target" != "$detected_target" ]]; then
  printf '%s\n' 'runner-gpu: target mismatch' >&2
  exit 2
fi

case "$target" in
  a) compile_env=ampere; target_code=a ;;
  p) compile_env=blackwell; target_code=p ;;
  *) printf '%s\n' 'target: invalid' >&2; exit 2 ;;
esac

short_head="${head_sha:0:12}"
short_tree="${tree_sha:0:12}"
if [[ "$release_mode" == true ]]; then
  [[ "$image_ref_override" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] || {
    echo "release image reference is invalid" >&2
    exit 2
  }
  image_ref="$image_ref_override"
  tag="${image_ref##*:}"
else
  tag="${CANDIDATE_TAG_PREFIX:-pr}-${pr_number}-head-${short_head}-${target_code}-${short_tree}"
  image_ref="${CI_CANDIDATE_IMAGE_REPOSITORY:-loongforge-ci}:$tag"
fi
docker_bin="${DOCKER_BIN:-docker}"

candidate_retention_hours="${LOONGFORGE_CANDIDATE_RETENTION_HOURS:-24}"
[[ "$candidate_retention_hours" =~ ^[0-9]+$ ]] || {
  echo "LOONGFORGE_CANDIDATE_RETENTION_HOURS must be an integer" >&2
  exit 2
}

trusted_dockerfile="$script_dir/../../../docker/Dockerfile"
source_dockerfile="${IMAGE_DOCKERFILE:-$trusted_dockerfile}"
[[ -f "$source_dockerfile" ]] || { printf '%s\n' 'dockerfile: missing' >&2; exit 2; }
context_dir="$(mktemp -d)"
mkdir -p "$context_dir/LoongForge"
tar -C "$source_dir" \
  --exclude=.git \
  --exclude=.pytest_cache \
  --exclude='*.log' \
  --exclude='__pycache__' \
  -cf - . | tar -C "$context_dir/LoongForge" -xf -
trap 'rm -rf "$context_dir"' EXIT
dockerfile="$source_dockerfile"

build_args=(
  --network host
  --build-arg "COMPILE_ENV=$compile_env"
  --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  --label "org.opencontainers.image.revision=$head_sha"
  --label "io.loongforge.tree-sha=$tree_sha"
  --label "io.loongforge.image-target=$target_code"
  --label "io.loongforge.image-revision=$image_ref"
)
if [[ "$release_mode" == true ]]; then
  build_args+=(--label "io.loongforge.release=true")
else
  build_args+=(
    --label "io.loongforge.pr-number=$pr_number"
    --label "io.loongforge.candidate=true"
  )
fi
[[ -n "${IMAGE_BASE_IMAGE:-}" ]] && build_args+=(--build-arg "BASE_IMAGE=$IMAGE_BASE_IMAGE")
for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  proxy_value="${!proxy_var:-}"
  [[ -n "$proxy_value" ]] || continue
  if [[ "$proxy_value" =~ ://[^/[:space:]]*@ ]]; then
    printf '%s\n' 'proxy: embedded credentials are not allowed' >&2
    exit 2
  fi
  build_args+=(--build-arg "$proxy_var=$proxy_value")
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
  [[ -f "$secret_path" ]] || { echo "$config_var file not found" >&2; exit 2; }
  build_args+=(--secret "id=$secret_id,src=$secret_path")
done

# A cancelled workflow can bypass regression's normal cleanup. Restrict this
# prune to old, unused candidate images created by this CI. Release images are
# retained until the workflow's explicit cleanup step.
if [[ "$release_mode" == false ]]; then
  "$docker_bin" image prune -af \
    --filter "label=io.loongforge.candidate=true" \
    --filter "until=${candidate_retention_hours}h" >/dev/null 2>&1 || true
fi

redact_build_output() {
  python3 "$script_dir/../../../ci/redact_ci_artifact.py" --stream >&2
}

if ! DOCKER_BUILDKIT=1 "$docker_bin" build "${build_args[@]}" \
  -f "$dockerfile" -t "$image_ref" "$context_dir" 2>&1 | redact_build_output; then
  if [[ "$release_mode" == true ]]; then
    echo "release image build failed" >&2
  else
    echo "candidate image build failed" >&2
  fi
  exit 1
fi

if ! DOCKER_BIN="$docker_bin" python3 "$script_dir/../image_policy.py" "$image_ref" >&2; then
  "$docker_bin" image rm -f "$image_ref" >/dev/null 2>&1 || true
  if [[ "$release_mode" == true ]]; then
    echo "release image policy failed" >&2
  else
    echo "candidate image policy failed" >&2
  fi
  exit 1
fi

printf '%s\n' "$image_ref"
