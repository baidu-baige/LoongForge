mkdir -p /workspace/bridge_test_log/qwen/

sh tools/dist_checkpoint/test/qwen/1.8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.8b_log

sh tools/dist_checkpoint/test/qwen/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/7b_log

sh tools/dist_checkpoint/test/qwen/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/14b_log

sh tools/dist_checkpoint/test/qwen/72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/72b_log

sh tools/dist_checkpoint/test/qwen/1.5_7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_7b_log

sh tools/dist_checkpoint/test/qwen/1.5_72b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen/1.5_72b_log