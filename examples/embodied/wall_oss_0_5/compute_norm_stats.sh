#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Compute Wall-OSS-0.5 LeRobot Norm Stats (e.g. libero_norm_stats.json).
#
# Reads ``observation.state`` and ``action`` columns from a local LeRobot
# v2.1/v3 parquet dataset and writes per-dim mean/std/q01/q99 as JSON in the
# layout expected by the Wall-OSS-0.5 collator
# (see ``wall_oss_0_5/transforms/wall_oss_0_5_utils.py::load_norm_stats``).
#
# Usage:
#   bash compute_norm_stats.sh
#
# Environment variables (all optional, defaults shown):
#   LOONGFORGE_PATH   Path to the LoongForge repo root
#                     (default: /workspace/LoongForge)
#   DATASET_PATH      Root directory of the source LeRobot dataset
#                     (default: /path/to/libero)
#   OUTPUT_PATH       Output JSON path
#                     (default: $LOONGFORGE_PATH/data/wall_oss_0_5_norm_stats/libero_norm_stats.json)
#   STATE_KEY         Dataset column for proprioception (default: observation.state)
#   ACTION_KEY        Dataset column for action           (default: action)
#   PYTHON_BIN        Python interpreter to use           (default: python)
#
# Example:
#   DATASET_PATH=/datasets/libero \
#   OUTPUT_PATH=$LOONGFORGE_PATH/data/wall_oss_0_5_norm_stats/libero_norm_stats.json \
#   bash compute_norm_stats.sh
#
# Any extra arguments are forwarded to the Python script.

set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

DATASET_PATH="${DATASET_PATH:-/path/to/libero}"
OUTPUT_PATH="${OUTPUT_PATH:-$LOONGFORGE_PATH/data/wall_oss_0_5_norm_stats/libero_norm_stats.json}"
STATE_KEY="${STATE_KEY:-observation.state}"
ACTION_KEY="${ACTION_KEY:-action}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge Wall-OSS-0.5 Norm Stats"
echo "  Data:     $DATASET_PATH"
echo "  Output:   $OUTPUT_PATH"
echo "  Keys:     $STATE_KEY / $ACTION_KEY"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  "$PYTHON_BIN" "$LOONGFORGE_PATH/loongforge/embodied/data/datasets/wall_oss_0_5/compute_norm_stats.py" \
    --data-root "$DATASET_PATH" \
    --output-path "$OUTPUT_PATH" \
    --state-key "$STATE_KEY" \
    --action-key "$ACTION_KEY" \
    "$@"

echo "Wall-OSS-0.5 norm stats written to: $OUTPUT_PATH"
