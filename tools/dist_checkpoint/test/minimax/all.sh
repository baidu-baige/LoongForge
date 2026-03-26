mkdir -p /workspace/bridge_test_log/minimax/

sh tools/dist_checkpoint/test/minimax/m2_1_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/minimax/m2_1_log