mkdir -p /workspace/bridge_test_log/deepseek3/

sh tools/dist_checkpoint/test/deepseek3/v3_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek3/v3_log

sh tools/dist_checkpoint/test/deepseek3/v3_2_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek3/v3_2_log