#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-LLM"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen2-7B/"}

input_data=/mnt/cluster/aiak-training-llm/dataset/pile_test/train.jsonl
output_prefix=/mnt/cluster/aiak-training-llm/qwen2/pile_test/pile-qwen

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    python ${AIAK_TRAINING_PATH}/tools/data_preprocess/preprocess_pretrain_data.py \
        --input ${input_data} \
        --output-prefix ${output_prefix} \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --json-keys text \
        --workers 50 \
        --append-eod
