mkdir -p /workspace/bridge_test_log/llama3/

sh tests/mcore_bridge/llama3/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama3/8b_log

sh tests/mcore_bridge/llama3/70b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama3/70b_log