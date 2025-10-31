#!/bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
FP8_QUANTIZATION_PATH="$AIAK_TRAINING_PATH/tools/fp8_quantization"

# hf bf16 path
HF_BF16_CKPT_PATH=/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-bf16-hf
# hf fp8 path
HF_FP8_CKPT_PATH=/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-fp8-hf

python $FP8_QUANTIZATION_PATH/hf_bf16_to_fp8.py \
  --bf16-dir $HF_BF16_CKPT_PATH \
  --fp8-dir $HF_FP8_CKPT_PATH