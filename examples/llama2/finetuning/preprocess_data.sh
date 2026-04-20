#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/meta-llama/Llama-2-7b-hf/"}

input_data=/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json
output_path=/mnt/cluster/LoongForge/llama2/sft_aplaca_zh_tokenized

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
        --input ${input_data} \
        --output ${output_path} \
        --seq-length 2048 \
        --chat-template llama2 \
        --tokenizer-type HFTokenizer \
        --hf-tokenizer-path $TOKENIZER_PATH \
        --workers 50 \
        --split 100,0,0
        # --packing-sft-data \
        # --train-on-prompt \
        # --eod-mask-loss \
        # --sft-dataset-config /workspace/LoongForge/configs/data/sft_dataset_config.yaml \
        # --sft-dataset custom_dataset \
