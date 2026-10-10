# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/qwen2.5/

sh tests/mcore_bridge_roundtrip/qwen2.5/0.5b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/0.5b_log

sh tests/mcore_bridge_roundtrip/qwen2.5/1.5b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/1.5b_log

sh tests/mcore_bridge_roundtrip/qwen2.5/3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/3b_log

sh tests/mcore_bridge_roundtrip/qwen2.5/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/7b_log

sh tests/mcore_bridge_roundtrip/qwen2.5/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/14b_log

sh tests/mcore_bridge_roundtrip/qwen2.5/32b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5/32b_log
