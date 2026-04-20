#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen-7B/"}

input_data=/mnt/cluster/LoongForge/dataset/pile_test/train.jsonl
output_prefix=/mnt/cluster/LoongForge/qwen/pile_test/pile-qwen

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_pretrain_data.py \
        --model-family qwen \
        --input ${input_data} \
        --output-prefix ${output_prefix} \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --json-keys text \
        --workers 50 \
        --append-eod
