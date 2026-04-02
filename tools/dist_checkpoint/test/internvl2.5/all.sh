mkdir -p /workspace/bridge_test_log/internvl2.5/

sh tools/dist_checkpoint/test/internvl2.5/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl2.5/8b_log

sh tools/dist_checkpoint/test/internvl2.5/26b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl2.5/26b_log

sh tools/dist_checkpoint/test/internvl2.5/38b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl2.5/38b_log

sh tools/dist_checkpoint/test/internvl2.5/78b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl2.5/78b_log
