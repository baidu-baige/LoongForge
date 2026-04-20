#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
input_string=$1

export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron/"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

if [ "$input_string" == "hg2mcore" ]; then
    echo "convert weight from huggingface to megatron"
    # wan2.2 high noise model
    python ./convert_checkpoint_hg2mcore.py  --load_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/high_noise/Megatron_Random/iter_0000002" \
                                --save_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/high_noise/Megatron_Release/" \
                                --checkpoint_path="/mnt/cluster/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/" \
                                --num_checkpoints=6 \
                                --tp=1 \
                                --pp=4 \
                                --num_layers=40 \
                                --model_name="wan2_2_i2v"
    # wan 2.2 low noise model
    python ./convert_checkpoint_hg2mcore.py  --load_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/low_noise/Megatron_Random/iter_0000002" \
                                --save_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/low_noise/Megatron_Release/" \
                                --checkpoint_path="/mnt/cluster/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/" \
                                --num_checkpoints=6 \
                                --tp=1 \
                                --pp=4 \
                                --num_layers=40 \
                                --model_name="wan2_2_i2v"
elif [ "$input_string" == "mcore2hg" ]; then
    echo "convert weight from megatron to huggingface"
    # wan2.2 high noise model
    python ./convert_checkpoint_mcore2hg.py  --load_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/high_noise/Megatron_Release/" \
                                --save_path="/mnt/cluster/LoongForge/wan2.2/hg/high_noise/" \
                                --checkpoint_path="/mnt/cluster/models/Wan-AI/Wan2.2-I2V-A14B/high_noise_model/" \
                                --num_checkpoints=6 \
                                --tp=1 \
                                --pp=4 \
                                --num_layers=40 \
                                --model_name="wan2_2_i2v"
    # wan 2.2 low noise model
    python ./convert_checkpoint_mcore2hg.py  --load_path="/mnt/cluster/LoongForge/wan2.2/hg2mcore/low_noise/Megatron_Release/" \
                                --save_path="/mnt/cluster/LoongForge/wan2.2/hg/low_noise/" \
                                --checkpoint_path="/mnt/cluster/models/Wan-AI/Wan2.2-I2V-A14B/low_noise_model/" \
                                --num_checkpoints=6 \
                                --tp=1 \
                                --pp=4 \
                                --num_layers=40 \
                                --model_name="wan2_2_i2v"
else
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\" "
    exit 1
fi
