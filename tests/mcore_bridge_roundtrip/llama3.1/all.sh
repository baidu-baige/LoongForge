# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/llama3.1/

sh tests/mcore_bridge_roundtrip/llama3.1/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama3.1/8b_log

sh tests/mcore_bridge_roundtrip/llama3.1/70b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama3.1/70b_log

sh tests/mcore_bridge_roundtrip/llama3.1/405b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama3.1/405b_log