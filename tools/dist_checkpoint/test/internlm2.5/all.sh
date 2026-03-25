mkdir -p /workspace/bridge_test_log/internlm2.5/

sh tools/dist_checkpoint/test/internlm2.5/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internlm2.5/8b_log

sh tools/dist_checkpoint/test/internlm2.5/20b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internlm2.5/20b_log