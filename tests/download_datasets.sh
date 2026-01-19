#!/bin/bash

################################################################################################################################################################################
# Complete dataset download address reference: https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/AOhtOwLQrY/YCzcl7yVzd/6D6r9-PcDwNKfq#anchor-908b35c0-ffd6-11ed-b80d-c3136fa9d017
#
# Download location for dependent datasets:
# checkpoint dir：${PFS_DIR}/megatron_checkpoint
# tokenizers dir：${PFS_DIR}/tokenizers
# datasets dir：${PFS_DIR}/datasets

################################################################################################################################################################################

PFS_DIR="/ssd1/test"
# wget -P ${PFS_DIR} https://doc.bce.baidu.com/bos-optimization/linux-bcecmd-0.5.9.zip 
# unzip -o "${PFS_DIR}/linux-bcecmd-0.5.9.zip" -d "${PFS_DIR}"
# boscmd_dir="${PFS_DIR}/linux-bcecmd-0.5.9/bcecmd"
ln -s /ssd1/test/linux-bcecmd-0.3.9/bcecmd /usr/sbin/bcecmd

checkpoint_dir=${PFS_DIR}/megatron_checkpoint
huggingface_dir=${PFS_DIR}/huggingface.co
datasets_dir=${PFS_DIR}/omni_datasets/aiak-training-omni/

mkdir -p $checkpoint_dir $huggingface_dir $datasets_dir

# deepseek_v2_lite
bcecmd bos sync bos:/ai-data/deepseek-ai/DeepSeek-V2-Lite ${huggingface_dir}/deepseek-ai/DeepSeek-V2-Lite
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/deepseek_v2_lite/pile_test/ ${datasets_dir}/deepseek_v2_lite/pile_test/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/deepseek_v2_lite/tp2pp1ep8/ ${checkpoint_dir}/megatron_checkpoint/deepseek_v2_lite/

# llama2_7b
bcecmd bos sync bos:/ai-data/Llama-2-7b-hf-test ${huggingface_dir}/meta-llama/Llama-2-7b-hf
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llama2/pile_test/ ${datasets_dir}/llama2/pile_test/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama2_7b/tp1pp1/ ${checkpoint_dir}/megatron_checkpoint/llama2_7b/

# llama3_8b
bcecmd bos sync bos:/ai-data/meta-llama/Meta-Llama-3-8B ${huggingface_dir}/meta-llama/Meta-Llama-3-8B
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama3_8b/tp2pp2/ ${checkpoint_dir}/megatron_checkpoint/llama3_8b/

# qwen3_14b
bcecmd bos sync bos:/ai-data/Qwen/Qwen3-14B ${huggingface_dir}/Qwen/Qwen3-14B
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen3/pile_test/ ${datasets_dir}/qwen3/pile_test/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_14b/tp2pp4/ ${checkpoint_dir}/megatron_checkpoint/qwen3_14b/

# qwen2.5_vl_7b
bcecmd bos sync bos:/ai-data/Qwen/Qwen2.5-VL-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-VL-7B-Instruct/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds ${datasets_dir}/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/sft/test_packing/test_vqa_16k_packed_wds ${datasets_dir}/qwen2.5_vl_7b/sft/test_packing/test_vqa_16k_packed_wds

# llavaov_1.5_4b
bcecmd bos sync bos:/ai-data/lmms-lab/LLaVA-OneVision-1.5-4B-stage0 ${huggingface_dir}/LLaVA-OneVision-1.5-4B-stage0/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llavaov_1.5_4b/imagenet/CN/part00/part_00_wds ${datasets_dir}/llavaov_1.5_4b/imagenet/CN/part00/part_00_wds

# internvl2.5_8b
bcecmd bos sync bos:/ai-data/InternVL2_5-8B ${huggingface_dir}/internvl/InternVL2_5-8B/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_image ${datasets_dir}/internvl2.5_8b/webdataset_image

# internvl3.5_30b_a3b
bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL3_5-30B-A3B ${huggingface_dir}/Internvl3.5_30b_a3b/
bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_video ${datasets_dir}/internvl3.5_30b_a3b/webdataset_video
