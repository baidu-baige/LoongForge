# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/mimo/

sh tests/mcore_bridge_roundtrip/mimo/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/mimo/7b_log