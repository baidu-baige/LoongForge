#!/bin/bash
set -e  # 任何命令失败立即退出

CONFIG="config.yaml"

echo "=== [Step 1] Compute Sample Length==="

python get_sample_len.py --config "${CONFIG}"

echo "=== [Step 2] Hash-Bucket Split==="

python do_hashbacket.py --config "${CONFIG}"

echo "=== [Step 3] Prepare Final Packed Samples==="


python prepare_raw_samples.py --config "${CONFIG}"

echo "=== [Step 4] Pack to WDS Format==="

python packed_to_wds.py --config "${CONFIG}"