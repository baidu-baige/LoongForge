#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
input_string=$1

MEGATRON_PATH=/workspace/ernie/AIAK-Megatron/
AIAK_TRAINING_PATH=/workspace/ernie/AIAK-Training-Omni/

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH 

if [ "$input_string" == "hg2mcore" ] || [ "$input_string" == "mcore2hg" ]; then
    python ./ernie4.5vl_hg2mcore.py  --load_mcore_path="/workspace/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_save/iter_0000002/" \
                                --save_mcore_path="/ssd1/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_hg2mcore/" \
                                --load_hg_path="/workspace/ernie/ERNIE-4.5-VL-28B-A3B-PT/" \
                                --save_hg_path="/ssd1/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-PT_mcore2hg/" \
                                --tp=1 \
                                --pp=8 \
                                --num_vit_layers=32 \
                                --num_lm_layers=28 \
                                --convert_mode=$input_string
else
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
