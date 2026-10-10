# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/qwen2.5vl/

sh tests/mcore_bridge_roundtrip/qwen2.5vl/3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/3b_log

sh tests/mcore_bridge_roundtrip/qwen2.5vl/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/7b_log

sh tests/mcore_bridge_roundtrip/qwen2.5vl/32b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/32b_log

sh tests/mcore_bridge_roundtrip/qwen2.5vl/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/72b_log
