#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Prepare LeRobot metadata required by DreamZero. This does not generate the
# optional frozen-feature cache; use precompute_dreamzero_cache.sh for that.

set -euo pipefail

# ── Paths and dataset ───────────────────────────
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
DREAMZERO_DATA_ROOT=${DREAMZERO_DATA_ROOT:-"/workspace/dreamzero/data"}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-"oxe_droid"}

case "$EMBODIMENT_TAG" in
    oxe_droid)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/droid_lerobot"
        ;;
    libero_sim)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/libero_lerobot"
        ;;
    agibot)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/agibot_lerobot"
        ;;
    yam)
        DEFAULT_DATA_PATH="$DREAMZERO_DATA_ROOT/yam_lerobot"
        ;;
    *)
        echo "Unsupported EMBODIMENT_TAG: $EMBODIMENT_TAG" >&2
        exit 1
        ;;
esac

DATA_PATH=${DATA_PATH:-"$DEFAULT_DATA_PATH"}

# ── Preparation ──────────────────────────────
# FORCE replaces existing DreamZero metadata instead of validating it.
FORCE=${FORCE:-0}
# SKIP_STATISTICS generates schema metadata only.
SKIP_STATISTICS=${SKIP_STATISTICS:-0}
# Values <= 0 use all episodes for relative-action statistics.
MAX_RELATIVE_STAT_EPISODES=${MAX_RELATIVE_STAT_EPISODES:-10000}

PREPARE_ARGS=(
    --dataset-path "$DATA_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --max-relative-stat-episodes "$MAX_RELATIVE_STAT_EPISODES"
)

if [[ "$FORCE" == "1" ]]; then
    PREPARE_ARGS+=(--force)
fi
if [[ "$SKIP_STATISTICS" == "1" ]]; then
    PREPARE_ARGS+=(--skip-statistics)
fi

echo "============================================================"
echo "  LoongForge DreamZero Dataset Preparation"
echo "  Dataset:     $DATA_PATH"
echo "  Embodiment:  $EMBODIMENT_TAG"
echo "============================================================"

python "$LOONGFORGE_PATH/tools/data_preprocess/embodied/dreamzero/prepare_dataset.py" \
    "${PREPARE_ARGS[@]}" \
    "$@"
