#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Embodied regression entry point (runs inside the container).
#
# This wrapper only sources config/env.sh — all parameter parsing, validation,
# defaults, and optional --prepare execution live in cli.py.
#
# Both env-var and CLI-flag forms are supported (both handled by cli.py):
#   chip=A800 model_names="pi05_ddp groot_n1_6_ddp" bash tests/embodied/run.sh
#   bash tests/embodied/run.sh --chip A800 --models pi05_ddp groot_n1_6_ddp --fail_fast
#
# Calling cli.py directly also works; env.sh will be auto-loaded on first use.
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Source env.sh so path variables (EMBODIED_CI_ROOT / LOCAL_VLA_ARTIFACTS_ROOT / ...)
# propagate to both cli.py and the training subprocesses it launches.
source "${SCRIPT_DIR}/config/env.sh"

exec python3 "${SCRIPT_DIR}/cli.py" "$@"
