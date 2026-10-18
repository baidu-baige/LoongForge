#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
redactor="$script_dir/../../../ci/redact_ci_artifact.py"

"${DOCKER_BIN:-docker}" "$@" 2>&1 | python3 "$redactor" --stream
status="${PIPESTATUS[0]}"
exit "$status"
