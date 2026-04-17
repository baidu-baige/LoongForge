#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/meta-llama/Llama-2-7b-hf/"}

input_data=/mnt/cluster/BaigeOmni/dataset/sft_aplaca_zh_data.json
output_path=/mnt/cluster/BaigeOmni/llama2/sft_aplaca_zh_tokenized

PYTHONPATH=$MEGATRON_PATH:$BAIGE_OMNI_PATH:$PYTHONPATH \
    python ${BAIGE_OMNI_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
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
        # --sft-dataset-config /workspace/BaigeOmni/configs/data/sft_dataset_config.yaml \
        # --sft-dataset custom_dataset \
