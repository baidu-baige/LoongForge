#!/bin/bash

################################################################################################################################################################################
# 完整的数据集下载地址参考: https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/AOhtOwLQrY/YCzcl7yVzd/6D6r9-PcDwNKfq#anchor-908b35c0-ffd6-11ed-b80d-c3136fa9d017
# 
# 依赖数据集的下载位置：
# checkpoint dir：${PFS_DIR}/megatron_checkpoint
# tokenizers dir：${PFS_DIR}/tokenizers
# datasets dir：${PFS_DIR}/datasets

################################################################################################################################################################################

PFS_DIR="/mnt/pfs/leoli"

boscmd_dir="./linux-bcecmd-0.3.9/bcecmd"
checkpoint_dir=${PFS_DIR}/megatron_checkpoint
tokenizers_dir=${PFS_DIR}/tokenizers
datasets_dir=${PFS_DIR}/datasets

mkdir -p $checkpoint_dir $tokenizers_dir $datasets_dir


# chatglm-6b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/chatglm-6b-meg/tp1-pp1-dp8 ${checkpoint_dir}/chatglm-6b-meg/tp1-pp1-dp8
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/chatglm-6b-tokenizers ${tokenizers_dir}/chatglm-6b-tokenizers
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/datasets/wudao ${datasets_dir}/wudao

# galactica_6.7b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/megatron_galactica_6.7b_checkpoint_tp1_pp1_dp8_zero1 ${checkpoint_dir}/galactica/megatron_galactica_6.7b_checkpoint_tp1_pp1_dp8_zero1
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/galactica-hf-config ${tokenizers_dir}/galactica/galactica-hf-config
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/pile_galactica_test ${datasets_dir}/galactica/pile_galactica_test

# galactica_30b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/megatron_galactica_30b_checkpoint_tp2_pp2_dp4_zero1 ${checkpoint_dir}/galactica/megatron_galactica_30b_checkpoint_tp2_pp2_dp4_zero1
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/galactica-hf-config ${tokenizers_dir}/galactica/galactica-hf-config
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/galactica/pile_galactica_test ${datasets_dir}/galactica/pile_galactica_test

# GLM 10B Chinese
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/glm-10b-ckpt ${checkpoint_dir}/glm-10b-ckpt
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/glm-10b-tokenizers ${tokenizers_dir}/glm-10b-tokenizers
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/datasets/cc_download_articles ${datasets_dir}/cc_download_articles

# llama_7b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/megatron_llama_7b_checkpoint_tp1_pp1_dp8_zero1 ${checkpoint_dir}/megatron_llama/megatron_llama_7b_checkpoint_tp1_pp1_dp8_zero1
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/llama_tokenizer ${tokenizers_dir}/megatron_llama/llama_tokenizer
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/pile_llama_test ${datasets_dir}/megatron_llama/pile_llama_test

# llama_13b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/megatron_llama_13b_checkpoint_tp2_pp1_dp4_zero1 ${checkpoint_dir}/megatron_llama/megatron_llama_13b_checkpoint_tp2_pp1_dp4_zero1
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/llama_tokenizer ${tokenizers_dir}/megatron_llama/llama_tokenizer
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/pile_llama_test ${datasets_dir}/megatron_llama/pile_llama_test

# aquila_7b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/aquilaChat-7b-tp8-pp1-dp1-fp32 ${checkpoint_dir}/aquilaChat-7b-tp8-pp1-dp1-fp32
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/aquilachat-7b ${tokenizers_dir}/aquilachat-7b
$boscmd_dir bos sync bos:/cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/aquilachat-7b ${datasets_dir}/aquilachat-7b
