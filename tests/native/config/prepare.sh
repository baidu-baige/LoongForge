#!/bin/bash
# Embodied regression environment preparation (idempotent):
#   Sync the entire vla_artifacts from the configured source to
#   ${LOCAL_VLA_ARTIFACTS_ROOT}.
#
# Usage:
#   bash config/prepare.sh
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${LOCAL_VLA_ARTIFACTS_ROOT}"

REAL_BCECMD=${BCECMD}
bcecmd() {
    local retries=20
    local count=0
    local wait_time=3
    while [ $count -lt $retries ]; do
        set +e
        "$REAL_BCECMD" "$@"
        local status=$?
        set -e
        if [ $status -eq 0 ]; then
            return 0
        fi
        count=$((count + 1))
        echo "[WARNING] bcecmd failed (exit=$status); retrying in ${wait_time}s ($count/$retries)"
        sleep "$wait_time"
    done
    echo "[ERROR] bcecmd still failed after $retries retries"
    exit 1
}

echo "==== Syncing ${BOS_VLA_ARTIFACTS_ROOT} to ${LOCAL_VLA_ARTIFACTS_ROOT} ===="
bcecmd bos sync "${BOS_VLA_ARTIFACTS_ROOT}" "${LOCAL_VLA_ARTIFACTS_ROOT}"

echo "======================================"
echo "Embodied regression environment preparation complete"
echo "======================================"
