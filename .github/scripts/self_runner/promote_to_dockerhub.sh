#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Pull, retag, and push a validated image on the dedicated publisher host.

set -euo pipefail
umask 077

internal_ref="${1:?internal image reference is required}"
dockerhub_ref="${2:?Docker Hub image reference is required}"
internal_host="${3:?internal registry host is required}"
dockerhub_host="${4:?Docker Hub registry host is required}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/load_release_config.sh"
source "$script_dir/load_release_credentials.sh" iregistry
source "$script_dir/load_release_credentials.sh" dockerhub

redactor="$script_dir/../redact_ci_artifact.py"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
remote_config=""
destination_created=false
stage=prepare
login_target=""
source_id=""
push_digest=""
registry_digest=""
push_log="$(mktemp)"
mkdir -p loongforge-release-artifacts
promotion_log=loongforge-release-artifacts/dockerhub-promotion.log
: > "$promotion_log"

remote() {
  ssh "${ssh_options[@]}" -n "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$@" \
    2> >(python3 "$redactor" --stream >&2)
}

docker_command() {
  printf 'HTTPS_PROXY=%q NO_PROXY=%q timeout --kill-after=30s %q env -u DOCKER_HOST -u DOCKER_CONTEXT DOCKER_CLI_EXPERIMENTAL=enabled docker --config %q' \
    "$LOONGFORGE_DOCKERHUB_PROXY" "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
    "${LOONGFORGE_DOCKERHUB_STAGE_TIMEOUT_SECONDS:-3600}" "$remote_config"
  printf ' %q' "$@"
}

logged() {
  stage="$1"
  local command="$2" started=$SECONDS status
  printf 'dockerhub-promotion: %s started\n' "$stage" | tee -a "$promotion_log"
  set +e
  if [[ "$stage" == login_* ]]; then
    ssh "${ssh_options[@]}" "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$command" 2>&1 |
      tee "$push_log" | python3 "$redactor" --stream | tee -a "$promotion_log"
  else
    remote "$command" 2>&1 | tee "$push_log" | python3 "$redactor" --stream | tee -a "$promotion_log"
  fi
  status="${PIPESTATUS[0]}"
  set -e
  printf 'dockerhub-promotion: %s finished in %ss (exit %s)\n' \
    "$stage" "$((SECONDS - started))" "$status" | tee -a "$promotion_log"
  return "$status"
}

cleanup() {
  local status=$? command archive
  trap - EXIT
  set +e
  if [[ "$remote_config" =~ ^/tmp/loongforge-publish[-.][A-Za-z0-9]+$ ]]; then
    for host in "$internal_host" "$dockerhub_host"; do
      remote "$(docker_command logout "$host")" >/dev/null 2>&1
    done
    if [[ "$destination_created" == true ]]; then
      remote "$(docker_command image rm "$dockerhub_ref")" >/dev/null 2>&1
    fi
    printf -v command 'find %q -mindepth 1 -depth -delete && rmdir %q' "$remote_config" "$remote_config"
    remote "$command" >/dev/null 2>&1
  fi
  python3 - "$status" "$stage" "$login_target" "$source_id" "$push_digest" "$registry_digest" <<'PY'
import json
import sys
from pathlib import Path
status, stage, login_target, source, pushed, registry = sys.argv[1:]
Path("loongforge-release-artifacts/dockerhub-promotion.result.json").write_text(
    json.dumps({"status": "passed" if status == "0" else "failed", "stage": stage,
                "login_target": login_target, "exit_code": int(status), "source_image_id": source,
                "push_digest": pushed, "registry_digest": registry}) + "\n"
)
PY
  if [[ -n "${LOONGFORGE_RELEASE_EVIDENCE_ROOT:-}" && "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ ]]; then
    archive="$LOONGFORGE_RELEASE_EVIDENCE_ROOT/$GITHUB_RUN_ID/loongforge-release-artifacts"
    if ! { mkdir -p "$archive" && cp "$promotion_log" \
        loongforge-release-artifacts/dockerhub-promotion.result.json "$archive/"; } >/dev/null 2>&1; then
      printf '%s\n' 'publisher: evidence archive unavailable' >&2
    fi
  fi
  unlink "$push_log"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

remote_config="$(remote 'umask 077; mktemp -d /tmp/loongforge-publish.XXXXXXXX')"
[[ "$remote_config" =~ ^/tmp/loongforge-publish[-.][A-Za-z0-9]+$ ]] || {
  echo 'publisher: invalid temporary configuration' >&2
  exit 2
}
if remote "$(docker_command image inspect "$dockerhub_ref")" >/dev/null 2>&1; then
  echo 'publisher: destination tag already exists locally; refusing to replace it' >&2
  exit 2
fi

login_remote_registry() {
  login_target="$1"
  stage="login_$login_target"
  local host="$2" username="$3" password="$4" attempt status
  for attempt in 1 2 3; do
    if printf '%s' "$password" | logged "$stage" \
        "$(docker_command login --username "$username" --password-stdin "$host")"; then
      login_target=""
      return 0
    else
      status=$?
    fi
    if [[ "$attempt" == 3 ]] || ! grep -Eqi \
        'timed out|timeout|deadline exceeded|request cancel[l]?ed|connection reset|connection refused|temporary failure|unexpected EOF|service unavailable|bad gateway' \
        "$push_log"; then
      return "$status"
    fi
    printf 'dockerhub-promotion: %s temporary network failure; retrying (%s/3)\n' \
      "$stage" "$((attempt + 1))" | tee -a "$promotion_log"
    sleep "$((attempt * 2))"
  done
}

login_remote_registry iregistry "$internal_host" "$IREGISTRY_USERNAME" "$IREGISTRY_PASSWORD"
login_remote_registry dockerhub "$dockerhub_host" "$DOCKERHUB_USERNAME" "$DOCKERHUB_TOKEN"

logged pull "$(docker_command pull "$internal_ref")"
source_id="$(remote "$(docker_command image inspect --format '{{.Id}}' "$internal_ref")")"
[[ "$source_id" =~ ^sha256:[a-f0-9]{64}$ ]]
logged tag "$(docker_command tag "$internal_ref" "$dockerhub_ref")"
destination_created=true
logged push "$(docker_command push "$dockerhub_ref")"
push_digest="$(sed -nE 's/.*digest: (sha256:[a-f0-9]{64}).*/\1/p' "$push_log" | tail -1)"
[[ "$push_digest" =~ ^sha256:[a-f0-9]{64}$ ]]

stage=verify
verify_started=$SECONDS
printf 'dockerhub-promotion: verify started\n' | tee -a "$promotion_log"
# Read the tag through the authenticated Registry API. Older Docker clients can
# fail manifest inspection even after their daemon successfully pushes an image.
# Credentials stay in the private configuration created by this promotion.
verify_script="$script_dir/verify_dockerhub_manifest.py"
[[ -f "$verify_script" ]] || {
  echo 'publisher: verifier script is unavailable' >&2
  exit 2
}
printf -v verify_command \
  'HTTPS_PROXY=%q NO_PROXY=%q timeout --kill-after=10s 180s python3 - %q %q %q %q' \
  "$LOONGFORGE_DOCKERHUB_PROXY" "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
  "$remote_config" "$dockerhub_ref" "$source_id" "$push_digest"
registry_digest="$(ssh "${ssh_options[@]}" "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$verify_command" \
  2> >(python3 "$redactor" --stream | tee -a "$promotion_log" >&2) < "$verify_script")"
[[ "$registry_digest" == "$push_digest" ]]
printf 'dockerhub-promotion: source image content and published digest verified\n' | tee -a "$promotion_log"
printf 'dockerhub-promotion: verify finished in %ss\n' "$((SECONDS - verify_started))" | tee -a "$promotion_log"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'verified=true\n' >> "$GITHUB_OUTPUT"
fi
