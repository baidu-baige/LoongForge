#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source_dir=""
config=""
target=a
pr_number=""
merge_sha=""
promote=false
build=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_dir="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --target) target="$2"; shift 2 ;;
    --pr) pr_number="$2"; shift 2 ;;
    --merge-sha) merge_sha="$2"; shift 2 ;;
    --promote) promote=true; shift ;;
    --promote-only) promote=true; build=false; shift ;;
    *) echo "unknown lifecycle argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$source_dir/.git" && -f "$config" && -n "$pr_number" ]] || {
  echo "usage: $0 --source DIR --config FILE --pr NUMBER [--target a|h|p] [--promote|--promote-only]" >&2
  exit 2
}

export CI_CONFIG_PATH="$config"
set -a
# shellcheck disable=SC1090
source "$config"
set +a

export SOURCE_DIR="$source_dir"
export PR_NUMBER="$pr_number"
export HEAD_SHA="$(git --git-dir="$source_dir/.git" --work-tree="$source_dir" rev-parse HEAD)"
export TREE_SHA="$(git --git-dir="$source_dir/.git" --work-tree="$source_dir" rev-parse 'HEAD^{tree}')"
export MERGE_SHA="${merge_sha:-$HEAD_SHA}"
export LOONGFORGE_IMAGE_BUILDER="${LOONGFORGE_IMAGE_BUILDER:-$source_dir/.github/scripts/self_runner/build_candidate_image.sh}"
export LOONGFORGE_IMAGE_PROMOTER="${LOONGFORGE_IMAGE_PROMOTER:-$source_dir/.github/scripts/self_runner/promote_candidate_image.sh}"

if [[ "$build" == true ]]; then
  output="$(mktemp)"
  trap 'rm -f "$output"' EXIT
  export GITHUB_OUTPUT="$output"
  "$source_dir/.github/scripts/build_candidate_image.sh" "$target"
  revision="$(sed -n 's/^revision=//p' "$output")"
  [[ -n "$revision" ]] || { echo "candidate revision was not produced" >&2; exit 1; }
  printf 'candidate=%s\nhead_sha=%s\ntree_sha=%s\n' "$revision" "$HEAD_SHA" "$TREE_SHA"
fi

if [[ "$promote" == true ]]; then
  "$source_dir/.github/scripts/promote_candidate_image.sh" "$target"
fi
