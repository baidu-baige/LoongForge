mkdir -p /workspace/bridge_test_log/qwen3_next/

sh tests/mcore_bridge/qwen3_next/80b_a3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3_next/80b_a3b_log