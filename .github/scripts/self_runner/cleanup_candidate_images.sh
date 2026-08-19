#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

target=all
retention_days="${IMAGE_CANDIDATE_RETENTION_DAYS:-14}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --retention-days) retention_days="$2"; shift 2 ;;
    *) echo "unknown cleanup argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$target" =~ ^(a|h|p|all)$ && "$retention_days" =~ ^[0-9]+$ ]] || {
  echo "cleanup requires target a|h|p|all and an integer retention period" >&2
  exit 2
}

config="${CI_CONFIG_PATH_IMAGE:-${CI_CONFIG_PATH:-}}"
[[ -f "$config" ]] || { echo "image config not found: $config" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$config"
set +a

registry="${IMAGE_REGISTRY:?IMAGE_REGISTRY is required}"
registry_url="${registry%/}"
[[ "$registry_url" == http://* || "$registry_url" == https://* ]] || registry_url="https://$registry_url"
curl_args=(-fsSL)
if [[ -n "${INTERNAL_REGISTRY_USERNAME:-}" || -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]]; then
  [[ -n "${INTERNAL_REGISTRY_USERNAME:-}" && -n "${INTERNAL_REGISTRY_PASSWORD:-}" ]] || {
    echo "registry username and password must be configured together" >&2
    exit 2
  }
  curl_args+=(-u "$INTERNAL_REGISTRY_USERNAME:$INTERNAL_REGISTRY_PASSWORD")
fi

targets=($target)
[[ "$target" == all ]] && targets=(a h p)
now_epoch="$(date +%s)"
max_age_seconds=$((retention_days * 86400))
prefix="${CANDIDATE_TAG_PREFIX:-pr}-"

for target_code in "${targets[@]}"; do
  repo_var="IMAGE_REPOSITORY_${target_code^^}"
  repository="${!repo_var:-}"
  [[ -n "$repository" ]] || continue
  [[ "$repository" != "$registry/"* ]] || repository="${repository#"$registry/"}"
  tags_json="$(curl "${curl_args[@]}" "$registry_url/v2/$repository/tags/list?n=10000")"
  while IFS= read -r tag; do
    [[ -n "$tag" ]] || continue
    manifest="$(curl "${curl_args[@]}" \
      -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
      "$registry_url/v2/$repository/manifests/$tag")"
    config_digest="$(grep -o '"digest"[[:space:]]*:[[:space:]]*"sha256:[a-f0-9]*"' <<<"$manifest" \
      | head -1 | cut -d '"' -f4)"
    [[ -n "$config_digest" ]] || { echo "cannot resolve config digest for $repository:$tag" >&2; continue; }
    image_config="$(curl "${curl_args[@]}" "$registry_url/v2/$repository/blobs/$config_digest")"
    created="$(grep -o '"org.opencontainers.image.created"[[:space:]]*:[[:space:]]*"[^"]*"' \
      <<<"$image_config" | head -1 | cut -d '"' -f4)"
    created_epoch="$(date -d "$created" +%s 2>/dev/null || true)"
    [[ "$created_epoch" =~ ^[0-9]+$ ]] || { echo "cannot resolve creation time for $repository:$tag" >&2; continue; }
    (( now_epoch - created_epoch > max_age_seconds )) || continue

    digest="$(curl "${curl_args[@]}" -I \
      -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
      "$registry_url/v2/$repository/manifests/$tag" \
      | tr -d '\r' | awk -F ': ' 'tolower($1) == "docker-content-digest" {print $2}')"
    [[ -n "$digest" ]] || { echo "cannot resolve manifest digest for $repository:$tag" >&2; continue; }
    if [[ "${IMAGE_CLEANUP_DRY_RUN:-false}" == true ]]; then
      echo "would delete $repository:$tag ($digest)"
    else
      curl "${curl_args[@]}" -X DELETE "$registry_url/v2/$repository/manifests/$digest" >/dev/null
      echo "deleted $repository:$tag ($digest)"
    fi
  done < <(grep -o "\"${prefix}[^\"]*\"" <<<"$tags_json" | tr -d '"')
done
