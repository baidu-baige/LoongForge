mkdir -p /workspace/bridge_test_log/llama2/

sh tools/dist_checkpoint/test/llama2/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/7b_log

sh tools/dist_checkpoint/test/llama2/13b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/13b_log

sh tools/dist_checkpoint/test/llama2/70b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llama2/70b_log