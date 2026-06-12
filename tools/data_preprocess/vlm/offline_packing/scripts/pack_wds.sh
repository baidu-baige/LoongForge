#!/bin/bash
set -e  # Exit immediately if any command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

CONFIG="${1:-configs/config.yaml}"

format_elapsed() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

run_step() {
    local step_name="$1"
    shift
    local start_time
    local end_time
    local elapsed
    local status

    echo "=== [${step_name}] START $(date '+%F %T') ==="
    start_time=$(date +%s)
    set +e
    "$@"
    status=$?
    set -e
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [ "${status}" -eq 0 ]; then
        echo "=== [${step_name}] DONE elapsed=$(format_elapsed "${elapsed}") (${elapsed}s) ==="
    else
        echo "=== [${step_name}] FAILED status=${status} elapsed=$(format_elapsed "${elapsed}") (${elapsed}s) ==="
    fi

    return "${status}"
}

pipeline_start=$(date +%s)

echo "=== [Offline Packing] START $(date '+%F %T') config=${CONFIG} ==="

run_step "Step 1: Scan WDS Manifest and Compute Sample Length" \
    python -m wds_pack.cli.scan_manifest --config "${CONFIG}"

run_step "Step 2: Hash-Bucket Split by Media Type" \
    python -m wds_pack.cli.pack_bins --config "${CONFIG}"

run_step "Step 3: Build Pack Plan" \
    python -m wds_pack.cli.build_plan --config "${CONFIG}"

run_step "Step 4: Pack to WDS Format" \
    python -m wds_pack.cli.write_wds --config "${CONFIG}"

pipeline_end=$(date +%s)
pipeline_elapsed=$((pipeline_end - pipeline_start))

echo "=== [Offline Packing] DONE elapsed=$(format_elapsed "${pipeline_elapsed}") (${pipeline_elapsed}s) ==="
