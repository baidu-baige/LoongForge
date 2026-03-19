#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V2-Lite"}

input_data=/mnt/cluster/BaigeOmni/dataset/pile_test/train.jsonl
output_prefix=/mnt/cluster/BaigeOmni/deepseek2/pile_test/pile-deepseek


PYTHONPATH=$MEGATRON_PATH:$BAIGE_OMNI_PATH:$PYTHONPATH \
    python ${BAIGE_OMNI_PATH}/tools/data_preprocess/llm/preprocess_pretrain_data.py \
        --input ${input_data} \
        --output-prefix ${output_prefix} \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --json-keys text \
        --workers 50 \
        --append-eod
