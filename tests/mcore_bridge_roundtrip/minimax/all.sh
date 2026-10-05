# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/minimax/

sh tests/mcore_bridge_roundtrip/minimax/m2_1_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/minimax/m2_1_log