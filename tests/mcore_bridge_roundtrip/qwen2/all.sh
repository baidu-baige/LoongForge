# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/qwen2/

sh tests/mcore_bridge_roundtrip/qwen2/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2/7b_log

sh tests/mcore_bridge_roundtrip/qwen2/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2/72b_log