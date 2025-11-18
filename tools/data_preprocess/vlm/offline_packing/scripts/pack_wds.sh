#!/bin/bash
set -e  # 任何命令失败立即退出

CONFIG="config.yaml"

echo "=== [Step 1] Compute Sample Length==="

python get_sample_len.py --config "${CONFIG}"

echo "=== [Step 2] Hash-Bucket Split==="

python do_hashbacket.py --config "${CONFIG}"

echo "=== [Step 3] Prepare Final Packed Samples==="


python prepare_raw_samples.py --config "${CONFIG}"


packed_json_dir=$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}'))['data']['packed_json_dir'])")

python ../convert_to_webdataset.py \
    --json_file ${packed_json_dir} \
    --image_dir /mnt/cluster/hejinhui/data/qianfan/filter_mmdu/mmdu-45k_pics/ \
    --media image \
    --output_dir /mnt/cluster/yxc/test_6/ \
    --maxcount 1000000 \
    --maxsize 100000000 \
    --message_key texts \
    --packed true \
    --sample_type packed_multi_mix_qa 
