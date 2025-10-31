#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
input_string=$1

export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron/"}
export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH 

if [ "$input_string" == "hg2mcore" ]; then
    echo "convert weight from huggingface to megatron"
    python ./convert_checkpoint_hg2mcore.py  --load_path="/mnt/cluster/aiak-training-llm/wan2.1/iter_0000005/" \
                                --save_path="/mnt/cluster/aiak-training-llm/wan2.1/hg2mcore/release/" \
                                --checkpoint_path="/ssd1/models/huggingface.co/Wan2.1-I2V-14B-480P/" \
                                --num_checkpoints=7 \
                                --tp=1 \
                                --pp=8 \
                                --num_layers=40
elif [ "$input_string" == "mcore2hg" ]; then
    echo "convert weight from megatron to huggingface"
    python ./convert_checkpoint_mcore2hg.py  --load_path="/mnt/cluster/aiak-training-llm/wan2.1/release/" \
                                --save_path="/mnt/cluster/aiak-training-llm/wan2.1/hg/" \
                                --checkpoint_path="/ssd1/models/huggingface.co/Wan2.1-I2V-14B-480P/" \
                                --num_checkpoints=7 \
                                --tp=1 \
                                --pp=8 \
                                --num_layers=40
else
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
