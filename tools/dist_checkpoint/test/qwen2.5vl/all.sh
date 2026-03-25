mkdir -p /workspace/bridge_test_log/qwen2.5vl/

sh tools/dist_checkpoint/test/qwen2.5vl/3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/3b_log

sh tools/dist_checkpoint/test/qwen2.5vl/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/7b_log

sh tools/dist_checkpoint/test/qwen2.5vl/32b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/32b_log

sh tools/dist_checkpoint/test/qwen2.5vl/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen2.5vl/72b_log
