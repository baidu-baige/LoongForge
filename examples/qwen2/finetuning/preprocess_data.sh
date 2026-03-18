#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/leoli/qwen/Qwen2-7B-HF"}

input_data=/mnt/cluster/BaigeOmni/dataset/sft_aplaca_zh_data.json
output_path=/mnt/cluster/BaigeOmni/qwen2/sft_aplaca_zh_tokenized

PYTHONPATH=$MEGATRON_PATH:$OMNI_PATH:$PYTHONPATH \
    python ${OMNI_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
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
        # --sft-dataset-config /workspace/BaigeOmni/configs/sft_dataset_config.json \
        # --sft-dataset custom_dataset \
