#!/bin/bash
# Embodied regression environment preparation (idempotent):
#   Sync the entire vla_artifacts from BOS to ${LOCAL_VLA_ARTIFACTS_ROOT}.
#
# Usage:
#   bash config/prepare.sh
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Centralized path configuration (in the same directory as this file): LOCAL_VLA_ARTIFACTS_ROOT / BCECMD and others all come from env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${LOCAL_VLA_ARTIFACTS_ROOT}"

# bcecmd wrapper function with retries (consistent with the main framework's tests/download_datasets.sh)
REAL_BCECMD=${BCECMD}
bcecmd() {
    local retries=20
    local count=0
    local wait_time=3
    while [ $count -lt $retries ]; do
        set +e
        $REAL_BCECMD "$@"
        local status=$?
        set -e
        if [ $status -eq 0 ]; then
            return 0
        fi
        count=$((count + 1))
        echo "[WARNING] bcecmd failed (exit=$status); retrying in ${wait_time}s ($count/$retries)"
        sleep $wait_time
    done
    echo "[ERROR] bcecmd still failed after $retries retries"
    exit 1
}

# The BOS source prefix is centrally configured in config/env.sh
echo "==== Syncing ${BOS_VLA_ARTIFACTS_ROOT} → ${LOCAL_VLA_ARTIFACTS_ROOT} ===="
bcecmd bos sync "${BOS_VLA_ARTIFACTS_ROOT}" "${LOCAL_VLA_ARTIFACTS_ROOT}"

echo "======================================"
echo "Embodied regression environment preparation complete"
echo "======================================"
