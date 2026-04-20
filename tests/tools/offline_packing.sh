#!/bin/bash
set -eo pipefail

# Accept config file path as argument
CONFIG="$1"

# Ensure LOONGFORGE_PATH environment variable exists
# if [ -z "${LOONGFORGE_PATH}" ]; then
#     echo "Error: LOONGFORGE_PATH is not set."
#     echo "It should process by run.sh"
#     exit 1
# fi

# Locate original python tool scripts directory
TOOLS_DIR="/workspace/LoongForge/tools/data_preprocess/vlm/offline_packing"

if [ ! -d "$TOOLS_DIR" ]; then
    echo "Error: Python scripts directory not found at $TOOLS_DIR"
    exit 1
fi

echo "============================================================"
echo "Running Offline Packing Pipeline (Test Custom Implementation)"
echo "Config File: $CONFIG"
echo "Tools Dir:   $TOOLS_DIR"
echo "============================================================"

# Execute 4 steps in order
# This decouples dependency on original shell script and allows flexible checkpoints in testing

echo ">>> [Step 1] Running get_sample_len.py..."
python "${TOOLS_DIR}/get_sample_len.py" --config "${CONFIG}"

echo ">>> [Step 2] Running do_hashbacket.py..."
python "${TOOLS_DIR}/do_hashbacket.py" --config "${CONFIG}"

echo ">>> [Step 3] Running prepare_raw_samples.py..."
python "${TOOLS_DIR}/prepare_raw_samples.py" --config "${CONFIG}"

echo ">>> [Step 4] Running packed_to_wds.py..."
python "${TOOLS_DIR}/packed_to_wds.py" --config "${CONFIG}"

echo "============================================================"
echo "Offline packing pipeline finished successfully."
echo "============================================================"
