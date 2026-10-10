# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/qwen/

sh tests/mcore_bridge_roundtrip/qwen/1.8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.8b_log

sh tests/mcore_bridge_roundtrip/qwen/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/7b_log

sh tests/mcore_bridge_roundtrip/qwen/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/14b_log

sh tests/mcore_bridge_roundtrip/qwen/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/72b_log

sh tests/mcore_bridge_roundtrip/qwen/1.5_7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_7b_log

sh tests/mcore_bridge_roundtrip/qwen/1.5_72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_72b_log