mkdir -p /workspace/bridge_test_log/internvl3.5/

sh tools/dist_checkpoint/test/internvl3.5/8b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl3.5/8b_log

sh tools/dist_checkpoint/test/internvl3.5/14b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl3.5/14b_log

sh tools/dist_checkpoint/test/internvl3.5/30b_a3b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl3.5/30b_a3b_log

sh tools/dist_checkpoint/test/internvl3.5/38b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl3.5/38b_log

sh tools/dist_checkpoint/test/internvl3.5/241b_a28b_bridge_roundtrip.sh 2>&1 | tee -a /workspace/bridge_test_log/internvl3.5/241b_a28b_log
