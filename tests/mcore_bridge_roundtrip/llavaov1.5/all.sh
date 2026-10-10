# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
mkdir -p /workspace/bridge_test_log/llavaov1.5/

sh tests/mcore_bridge_roundtrip/llavaov1.5/4b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llavaov1.5/4b_log