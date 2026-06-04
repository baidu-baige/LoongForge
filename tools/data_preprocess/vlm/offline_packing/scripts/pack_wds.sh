#!/bin/bash
set -e  # Exit immediately if any command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

CONFIG="${1:-configs/config.yaml}"

echo "=== [Step 1] Scan WDS Manifest and Compute Sample Length ==="

python -m wds_pack.cli.scan_manifest --config "${CONFIG}"

echo "=== [Step 2] Hash-Bucket Split by Media Type ==="

python -m wds_pack.cli.pack_bins --config "${CONFIG}"

echo "=== [Step 3] Build Pack Plan ==="

python -m wds_pack.cli.build_plan --config "${CONFIG}"

echo "=== [Step 4] Pack to WDS Format==="

python -m wds_pack.cli.write_wds --config "${CONFIG}"
