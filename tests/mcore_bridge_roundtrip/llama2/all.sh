# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/llama2/

sh tests/mcore_bridge_roundtrip/llama2/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/7b_log

sh tests/mcore_bridge_roundtrip/llama2/13b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/13b_log

sh tests/mcore_bridge_roundtrip/llama2/70b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/70b_log