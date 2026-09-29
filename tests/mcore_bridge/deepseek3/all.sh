mkdir -p /workspace/bridge_test_log/deepseek3/

sh tests/mcore_bridge/deepseek3/v3_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek3/v3_log

sh tests/mcore_bridge/deepseek3/v3_2_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek3/v3_2_log