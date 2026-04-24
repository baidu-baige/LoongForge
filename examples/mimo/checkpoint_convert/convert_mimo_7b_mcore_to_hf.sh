#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/mimo/mimo_7b.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/mimo/ckpt_convert/mimo_convert.yaml

LOAD=/models/ckpt/XiaomiMiMo/MiMo-7B-SFT-tp1pp2/release
SAVE=/models/ckpt/XiaomiMiMo/MiMo-7B-SFT-hf/

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=2 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --max_workers=32