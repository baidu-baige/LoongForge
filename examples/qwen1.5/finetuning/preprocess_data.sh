#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-LLM"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen1.5-7B-Chat/"}

input_data=/mnt/cluster/aiak-training-llm/dataset/sft_aplaca_zh_data.json
output_path=/mnt/cluster/aiak-training-llm/qwen1.5/sft_aplaca_zh_tokenized

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    python ${AIAK_TRAINING_PATH}/tools/data_preprocess/preprocess_sft_data.py \
        --input ${input_data} \
        --output ${output_path} \
        --seq-length 2048 \
        --chat-template qwen \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --workers 50 \
        --split 100,0,0
        # --packing-sft-data \
        # --train-on-prompt \
        # --eod-mask-loss \
        # --sft-dataset-config /workspace/AIAK-Training-LLM/configs/sft_dataset_config.json \
        # --sft-dataset custom_dataset \
