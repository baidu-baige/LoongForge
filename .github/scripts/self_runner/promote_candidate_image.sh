#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target=""
pr_number=""
head_sha=""
merge_sha=""
tree_sha=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --pr) pr_number="$2"; shift 2 ;;
    --head-sha) head_sha="$2"; shift 2 ;;
    --merge-sha) merge_sha="$2"; shift 2 ;;
    --tree-sha) tree_sha="$2"; shift 2 ;;
    *) echo "unknown promoter argument: $1" >&2; exit 2 ;;
  esac
done

case "$target" in
  a|ampere) target_code=a ;;
  h|hopper) target_code=h ;;
  p|blackwell) target_code=p ;;
  all)
    promoted=0
    for image_target in a h p; do
      set +e
      IMAGE_ALLOW_MISSING=true "$0" --target "$image_target" --pr "$pr_number" \
        --head-sha "$head_sha" --merge-sha "$merge_sha" --tree-sha "$tree_sha"
      status=$?
      set -e
      case "$status" in
        0) promoted=$((promoted + 1)) ;;
        3) ;;
        *) exit "$status" ;;
      esac
    done
    echo "promoted image targets: $promoted" >&2
    exit 0
    ;;
  *) echo "unsupported image target: $target" >&2; exit 2 ;;
esac

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
if [[ -z "$repository" ]]; then
  echo "$repo_var is not configured" >&2
  [[ "${IMAGE_ALLOW_MISSING:-false}" == true ]] && exit 3
  exit 2
fi
if [[ -n "${INTERNAL_REGISTRY_USERNAME:-}" || -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]]; then
  [[ -n "$registry" && -n "${INTERNAL_REGISTRY_USERNAME:-}" && -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]] || {
    echo "registry host, username and password must be configured together" >&2
    exit 2
  }
  printf '%s' "$INTERNAL_REGISTRY_PASSWORD" | docker login "$registry" \
    --username "$INTERNAL_REGISTRY_USERNAME" --password-stdin >/dev/null
fi
if [[ -n "$registry" && "$repository" != "$registry/"* ]]; then
  image_repository="$registry/$repository"
else
  image_repository="$repository"
fi

short_head="${head_sha:0:12}"
short_tree="${tree_sha:0:12}"
tag="${CANDIDATE_TAG_PREFIX:-pr}-${pr_number}-head-${short_head}-${target_code}-${short_tree}"
candidate_ref="$image_repository:$tag"
default_var="IMAGE_DEFAULT_TAG_${target_code^^}"
default_tag="${!default_var:-default}"
default_ref="$image_repository:$default_tag"

if [[ "${IMAGE_DRY_RUN:-false}" == true ]]; then
  printf 'candidate=%s\ndefault=%s\nmerge_sha=%s\n' "$candidate_ref" "$default_ref" "$merge_sha"
  exit 0
fi

candidate_available=false
if [[ "${IMAGE_LOCAL_ONLY:-false}" == true ]]; then
  docker image inspect "$candidate_ref" >/dev/null 2>&1 && candidate_available=true
elif docker pull "$candidate_ref" >&2; then
  candidate_available=true
fi
if [[ "$candidate_available" != true ]]; then
  echo "candidate image not found: $candidate_ref" >&2
  [[ "${IMAGE_ALLOW_MISSING:-false}" == true ]] && exit 3
  exit 1
fi
labels="$(docker image inspect --format '{{json .Config.Labels}}' "$candidate_ref")"
for expected in \
  "org.opencontainers.image.revision=$head_sha" \
  "io.loongforge.pr-number=$pr_number" \
  "io.loongforge.tree-sha=$tree_sha" \
  "io.loongforge.image-target=$target_code"; do
  key="${expected%%=*}"
  value="${expected#*=}"
  grep -Fq "\"$key\":\"$value\"" <<<"$labels" || {
    echo "candidate label mismatch: $expected" >&2
    exit 1
  }
done

old_image=""
if [[ "${IMAGE_LOCAL_ONLY:-false}" == true ]]; then
  old_image="$(docker image inspect --format '{{.Id}}' "$default_ref" 2>/dev/null || true)"
elif docker pull "$default_ref" >&2; then
  old_image="$(docker image inspect --format '{{.Id}}' "$default_ref")"
fi
docker tag "$candidate_ref" "$default_ref"
if [[ "${IMAGE_LOCAL_ONLY:-false}" == true ]]; then
  printf 'promoted=%s\n' "$default_ref"
  exit 0
elif docker push "$default_ref" >&2; then
  printf 'promoted=%s\n' "$default_ref"
  exit 0
fi

if [[ -n "$old_image" ]]; then
  docker tag "$old_image" "$default_ref"
  docker push "$default_ref" >&2 || true
fi
echo "default tag update failed; rollback attempted" >&2
exit 1
