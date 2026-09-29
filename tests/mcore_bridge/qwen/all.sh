mkdir -p /workspace/bridge_test_log/qwen/

sh tests/mcore_bridge/qwen/1.8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.8b_log

sh tests/mcore_bridge/qwen/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/7b_log

sh tests/mcore_bridge/qwen/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/14b_log

sh tests/mcore_bridge/qwen/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/72b_log

sh tests/mcore_bridge/qwen/1.5_7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_7b_log

sh tests/mcore_bridge/qwen/1.5_72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_72b_log