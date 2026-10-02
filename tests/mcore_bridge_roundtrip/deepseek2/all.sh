mkdir -p /workspace/bridge_test_log/deepseek2/

sh tests/mcore_bridge_roundtrip/deepseek2/v2_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek2/v2_log

sh tests/mcore_bridge_roundtrip/deepseek2/v2_lite_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/deepseek2/v2_lite_log