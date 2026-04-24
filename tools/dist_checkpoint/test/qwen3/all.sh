mkdir -p /workspace/bridge_test_log/qwen3/

sh tools/dist_checkpoint/test/qwen3/0.6b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/0.6b_log

sh tools/dist_checkpoint/test/qwen3/1.7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/1.7b_log

sh tools/dist_checkpoint/test/qwen3/4b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/4b_log

sh tools/dist_checkpoint/test/qwen3/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/8b_log

sh tools/dist_checkpoint/test/qwen3/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/14b_log

sh tools/dist_checkpoint/test/qwen3/30b_a3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/30b_a3b_log

sh tools/dist_checkpoint/test/qwen3/32b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/32b_log

sh tools/dist_checkpoint/test/qwen3/235b_a22b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/235b_a22b_log

sh tools/dist_checkpoint/test/qwen3/480b_a35b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/480b_a35b_log

sh tools/dist_checkpoint/test/qwen3/coder_30b_a3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/qwen3/coder_30b_a3b_log