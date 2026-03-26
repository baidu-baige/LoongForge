mkdir -p /workspace/bridge_test_log/llavaov1.5/

sh tools/dist_checkpoint/test/llavaov1.5/4b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/llavaov1.5/4b_log