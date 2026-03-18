#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/meta-llama/Meta-Llama-3-8B/"}

input_data=/mnt/cluster/BaigeOmni/dataset/pile_test/train.jsonl
output_prefix=/mnt/cluster/BaigeOmni/llama3/pile_test/pile-llama

PYTHONPATH=$MEGATRON_PATH:$OMNI_PATH:$PYTHONPATH \
    python ${OMNI_PATH}/tools/data_preprocess/llm/preprocess_pretrain_data.py \
        --input ${input_data} \
        --output-prefix ${output_prefix} \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --json-keys text \
        --workers 50 \
        --append-eod
