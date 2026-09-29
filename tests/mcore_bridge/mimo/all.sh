mkdir -p /workspace/bridge_test_log/mimo/

sh tests/mcore_bridge/mimo/7b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/mimo/7b_log