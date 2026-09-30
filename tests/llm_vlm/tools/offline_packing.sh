#!/bin/bash
set -eo pipefail

# Accept config file path as argument
CONFIG="$1"

# Resolve CONFIG to an absolute path before we cd into the tool dir, so a
# relative config path passed by the harness keeps resolving.
if [ -n "${CONFIG}" ] && [ -f "${CONFIG}" ]; then
    CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi

# Locate the offline_packing tool package (contains wds_pack/).
# `python -m wds_pack.cli.*` requires this dir to be the working directory.
TOOLS_DIR="/workspace/LoongForge/tools/vlm_data_preprocess/offline_packing"

if [ ! -d "$TOOLS_DIR" ]; then
    echo "Error: offline_packing tool directory not found at $TOOLS_DIR"
    exit 1
fi

cd "${TOOLS_DIR}"

echo "============================================================"
echo "Running Offline Packing Pipeline (WDS-native wds_pack CLI)"
echo "Config File: $CONFIG"
echo "Tools Dir:   $TOOLS_DIR"
echo "============================================================"

# Execute the 4 WDS-native steps in order (mirrors scripts/pack_wds.sh).
# Kept as explicit per-step invocations so testing can checkpoint between them.

echo ">>> [Step 1] wds_pack.cli.scan_manifest (scan WDS + compute sample length)..."
python -m wds_pack.cli.scan_manifest --config "${CONFIG}"

echo ">>> [Step 2] wds_pack.cli.pack_bins (hash-bucket split by media type)..."
python -m wds_pack.cli.pack_bins --config "${CONFIG}"

echo ">>> [Step 3] wds_pack.cli.build_plan (build pack plan)..."
python -m wds_pack.cli.build_plan --config "${CONFIG}"

echo ">>> [Step 4] wds_pack.cli.write_wds (pack to WDS format)..."
python -m wds_pack.cli.write_wds --config "${CONFIG}"

echo "============================================================"
echo "Offline packing pipeline finished successfully."
echo "============================================================"
