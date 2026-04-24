mkdir -p /workspace/bridge_test_log/deepseek2/

sh tools/dist_checkpoint/test/deepseek2/v2_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek2/v2_log

sh tools/dist_checkpoint/test/deepseek2/v2_lite_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek2/v2_lite_log