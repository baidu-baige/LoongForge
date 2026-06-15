#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# messages_to_wds.sh : convert messages-format JSONL to WebDataset tar shards.
set -euo pipefail

INPUT="${INPUT:-/path/to/load/messages-format-jsonl}"   # messages-format JSONL
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/save/wds}"    # WDS output dir for kimi model
LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"
MAXCOUNT="${MAXCOUNT:-10000}"
MAXSIZE="${MAXSIZE:-3000000000}"

CONVERT_SCRIPT="$LOONGFORGE_PATH/tools/data_preprocess/vlm/convert_to_webdataset.py"
mkdir -p "$OUTPUT_DIR"

python3 "$CONVERT_SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --json_file  "$INPUT" \
    --media mix \
    --sample_type multi_mix_qa \
    --columns_messages messages \
    --maxcount "$MAXCOUNT" \
    --maxsize  "$MAXSIZE"

echo "[messages_to_wds] done -> $OUTPUT_DIR"
