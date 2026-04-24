#!/bin/bash
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-"python3"}

LOAD=/workspace/pi05_huggingface/
SAVE=/workspace/pi05_omni/
KEEP_PT=${KEEP_PT:-"false"}

EXTRA_ARGS=()
if [[ "${KEEP_PT}" == "true" ]]; then
    EXTRA_ARGS+=("--keep-pt")
fi

mkdir -p "${SAVE}/release"

"${PYTHON_BIN}" "${LOONGFORGE_PATH}/tools/convert_checkpoint/pi05/convert_hf_to_Mfsdp.py" \
    --model-dir "${LOAD}" \
    --dcp-output-dir "${SAVE}/release" \
    "${EXTRA_ARGS[@]}"

echo release > "${SAVE}/latest_checkpointed_iteration.txt"
