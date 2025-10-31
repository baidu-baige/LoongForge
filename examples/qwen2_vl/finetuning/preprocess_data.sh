#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen2-VL-2B-Instruct/"}

input_data=/mnt/cluster/aiak-training-llm/dataset/mllm/demo/mllm_demo.json
output_path=/mnt/cluster/aiak-training-llm/dataset/mllm/demo/sft_aplaca_zh_tokenized

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    python ${AIAK_TRAINING_PATH}/tools/data_preprocess/preprocess_sft_data.py \
        --input ${input_data} \
        --output ${output_path} \
        --seq-length 1024 \
        --enable-discard-sample \
        --chat-template qwen2-vl \
        --sft-dataset multimodal \
        --tokenizer-type HFTokenizer \
        --image-resolution 512 \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --workers 50 \
        --split 100,0,0
