mkdir -p /workspace/bridge_test_log/qwen2/

sh tools/dist_checkpoint/test/qwen2/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2/7b_log

sh tools/dist_checkpoint/test/qwen2/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2/72b_log