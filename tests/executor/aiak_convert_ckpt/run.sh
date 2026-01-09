#!/bin/bash
set -x
set -eo pipefail

############################################ 模型训练参数 ############################################

AIAK_MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/ssd1/workspace/AIAK-Training-Omni"}
# AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH=${convert_checkpoint_path:-"$AIAK_TRAINING_PATH/tools/convert_checkpoint"}

export PYTHONPATH=$AIAK_MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

CONVERT_ARGS=(
    ${CONVERT_ARGS}
)

final_task=$((WORLD_SIZE - 1))

# 默认命令为空数组
commands=()

# VLM 模型类型检测函数
is_vlm_model() {
    local model=$1
    if [[ "${model}" =~ "qwen2.5_vl" ]] || [[ "${model}" =~ "qwen2_vl" ]] || \
       [[ "${model}" =~ "internvl" ]] || [[ "${model}" =~ "llavaov" ]] || \
       [[ "${model}" =~ "qwen3_vl" ]]; then
        return 0
    fi
    return 1
}

# MoE 模型类型检测函数
is_moe_model() {
    local model=$1
    if [[ "${model}" =~ "a3b" ]] || [[ "${model}" =~ "moe" ]] || \
       [[ "${model}" =~ "deepseek_v2" ]] || [[ "${model}" =~ "deepseek_v3" ]]; then
        return 0
    fi
    return 1
}

# 如果是 VLM 模型（qwen2.5_vl, internvl, llavaov 等），使用新的 module_convertor 方式
if is_vlm_model "${model_name}"; then
    LANGUAGE_MODEL_ARGS=(
        ${LANGUAGE_MODEL_ARGS}
    )
    VISION_MODEL_ARGS=(
        ${VISION_MODEL_ARGS}
    )
    ADAPTER_ARGS=(
        ${ADAPTER_ARGS}
    )
    PATCH_ARGS=(
        ${PATCH_ARGS}
    )
    MERGE_ARGS=(
        ${MERGE_ARGS}
    )

    # 根据模型类型选择 adapter 转换脚本
    if [[ "${model_name}" =~ "internvl" ]]; then
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py"
    else
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py"
    fi

    # 根据模型类型选择 merge 脚本（MoE 模型使用 merge_megatron_expert.py）
    if is_moe_model "${model_name}"; then
        MERGE_SCRIPT="$CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py"
    else
        MERGE_SCRIPT="$CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py"
    fi

    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${LANGUAGE_MODEL_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${VISION_MODEL_ARGS[*]}"
        "python $ADAPTER_SCRIPT ${ADAPTER_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py ${PATCH_ARGS[*]}"
        "python $MERGE_SCRIPT ${MERGE_ARGS[*]}"
    )
    mkdir -p $CHECKPOINT_PATH
    echo release > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt
    
    # cleanup intermediate files
    commands+=(
        "rm -rf ${LANGUAGE_MODEL_PATH}"
        "rm -rf ${VISION_MODEL_PATH}"
        "rm -rf ${ADAPTER_PATH}"
        "rm -rf ${PATCH_PATH}"
    )
else
    # LLM 模型使用默认转换方式
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${CONVERT_ARGS[*]}"
    )
    mkdir -p $CHECKPOINT_PATH
    echo release > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt
fi


# 判断只能是一个master或者最后一个worker 运行
if [[ "${RANK}" != "" ]] && [[ "${RANK}" == "${final_task}" ]] && [[ "${dry_run}" != "true" ]]; then
    echo ""
    # 遍历命令数组并执行每个命令
    for command in "${commands[@]}"; do
        echo "执行命令: \"$command\""
        eval "$command"
    done
else
    echo "跳过当前节点的任务【当且仅当只有 是一个master 或者最后一个 worker 节点进行权重转化 !!!】"
fi