mkdir -p /workspace/bridge_test_log/llavaov1.5/

sh tests/mcore_bridge/llavaov1.5/4b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llavaov1.5/4b_log