#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

#export TEST_MODEL=qwen2
#export TEST_MODEL=mimo
#export TEST_MODEL=qwen3moe
#export TEST_MODEL=qwen2.5_vl_3b
export TEST_MODEL=qwen3_vl_30b_a3b
if [ $TEST_MODEL == "mimo" ]; then
    export TP_SIZE=2
    export PP_SIZE=2
    export VPP_SIZE=1
    export TP_RANKS=0,1
    export PP_RANKS=1
    export MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/mimo/mimo_7b.yaml
    export CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/mimo/ckpt_convert/mimo_convert.yaml
    export LOAD=/models/ckpt/XiaomiMiMo/MiMo-7B-SFT/
    export SAVE=/models/ckpt/XiaomiMiMo/MiMo-7B-SFT-tp2pp2/
fi

if [ $TEST_MODEL == "qwen3moe" ]; then
    export TP_SIZE=2
    export PP_SIZE=2
    export VPP_SIZE=1
    export TP_RANKS=0,1
    export PP_RANKS=0,1
    export EP_SIZE=8
    export ETP_SIZE=1
    export EP_RANKS=0,1,2,3,4,5,6,7
    export ETP_RANKS=0
    export MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/qwen3_coder_30b_a3b.yaml
    export CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert.yaml
    export LOAD=/models/ckpt/Qwen3-Coder-30B-A3B-Instruct
    export SAVE=/models/ckpt/Qwen3-Coder-30B-A3B-Instruct-tp2pp2ep8
fi

if [ $TEST_MODEL == "qwen2" ]; then
    export MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/qwen2_5_0_5b.yaml
    export CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert_llm.yaml
    export LOAD=/models/ckpt/Qwen2.5-0.5B-Instruct
fi

if [ $TEST_MODEL == "qwen2.5_vl_3b" ]; then
    export TP_SIZE=4
    export ENCODER_TP_SIZE=2
    export PP_SIZE=4
    export VPP_SIZE=1
    export TP_RANKS=0,1,2,3
    export PP_RANKS=0,1,2,3
    export MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_3b.yaml
    export CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml
    export VISION_PATCH_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml
    export ADAPTER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml
    export LOAD=/models/ckpt/Qwen2.5-VL-3B-Instruct
    export SAVE=/models/ckpt/Qwen2.5-VL-3B-Instruct-hf
fi

if [ $TEST_MODEL == "qwen3_vl_30b_a3b" ]; then
    export TP_SIZE=4
    export ENCODER_TP_SIZE=2
    export PP_SIZE=4
    export VPP_SIZE=1
    export TP_RANKS=0,1,2,3
    export PP_RANKS=0,1,2,3
    export EP_SIZE=8
    export ETP_SIZE=1
    export EP_RANKS=0,1,2,3,4,5,6,7
    export ETP_RANKS=0
    export MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml
    export CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_qwen3vl.yaml
    export VISION_PATCH_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen3_vit_convert.yaml
    export ADAPTER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_3_mlp_adapter_convert.yaml
    export LOAD=/models/ckpt/Qwen3-VL-30B-A3B-Instruct
    export SAVE=/models/ckpt/Qwen3-VL-30B-A3B-Instruct-HF
fi

#PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH/tools:$PYTHONPATH \
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$LOONGFORGE_PATH/tools:$PYTHONPATH \
    python $LOONGFORGE_PATH/tools/dist_checkpoint/test/test_hf_checkpoint_converter.py