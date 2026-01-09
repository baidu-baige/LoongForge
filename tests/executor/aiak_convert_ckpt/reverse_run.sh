#!/bin/bash
set -x
set -eo pipefail

############################################ mcore to hf checkpoint convert ############################################

AIAK_MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/ssd1/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH=${convert_checkpoint_path:-"$AIAK_TRAINING_PATH/tools/convert_checkpoint"}

export PYTHONPATH=$AIAK_MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

# 解析参数
REVERSE_KEY_ARGS=(
    ${REVERSE_KEY_ARGS}
)
REVERSE_LANGUAGE_MODEL_ARGS=(
    ${REVERSE_LANGUAGE_MODEL_ARGS}
)
REVERSE_VISION_MODEL_ARGS=(
    ${REVERSE_VISION_MODEL_ARGS}
)
REVERSE_ADAPTER_ARGS=(
    ${REVERSE_ADAPTER_ARGS}
)
REVERSE_PATCH_ARGS=(
    ${REVERSE_PATCH_ARGS}
)
REVERSE_MERGE_ARGS=(
    ${REVERSE_MERGE_ARGS}
)

final_task=$((WORLD_SIZE - 1))

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

# 默认命令为空数组
commands=()

# 只有 VLM 模型才进行逆向转换
if is_vlm_model "${model_name}"; then
    # 获取并行参数
    PP=${PIPELINE_MODEL_PARALLER_SIZE:-2}
    ETP=${ENCODER_TENSOR_MODEL_PARALLER_SIZE:-2}
    DTP=${DECODER_TENSOR_MODEL_PARALLER_SIZE:-2}
    
    # 根据模型类型选择 adapter 转换脚本
    if [[ "${model_name}" =~ "internvl" ]]; then
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py"
    else
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py"
    fi

    # Step 1: key reverser - 将omni格式的mcore权重转为标准mcore格式
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py ${REVERSE_KEY_ARGS[*]}"
    )
    
    # Step 2: language model mcore -> hf
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${REVERSE_LANGUAGE_MODEL_ARGS[*]}"
    )
    
    # Step 3: vision model mcore -> hf
    # 处理 PP > 1 的情况，需要创建临时目录
    if [[ $PP -gt 1 ]]; then
        # 允许通过环境变量覆盖加载路径，默认为 release
        MCORE_LOAD_DIR=${MCORE_LOAD_PATH:-"${CHECKPOINT_PATH}/release"}
        LOAD_PATH=${MCORE_LOAD_DIR}/tmp/

        commands+=(
            "mkdir -p $LOAD_PATH"
        )
        # 复制 vision model 相关的文件
        for ((i=0;i<$ETP;i++)); do
            from=$(printf "mp_rank_%02d_000" $i)
            to=$(printf "mp_rank_%02d" $i)
            commands+=(
                "cp -r ${MCORE_LOAD_DIR}/$from $LOAD_PATH/$to"
            )
        done
        # 更新 vision model 参数中的 load_ckpt_path
        REVERSE_VISION_MODEL_ARGS_MODIFIED=$(echo "${REVERSE_VISION_MODEL_ARGS[*]}" | sed "s|--load_ckpt_path=[^ ]*|--load_ckpt_path=$LOAD_PATH|g")
        commands+=(
            "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py $REVERSE_VISION_MODEL_ARGS_MODIFIED"
            "rm -rf $LOAD_PATH"
        )
    else
        commands+=(
            "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${REVERSE_VISION_MODEL_ARGS[*]}"
        )
    fi
    
    # Step 4: adapter mcore -> hf
    commands+=(
        "python $ADAPTER_SCRIPT ${REVERSE_ADAPTER_ARGS[*]}"
    )
    
    # Step 5: vision patch mcore -> hf
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py ${REVERSE_PATCH_ARGS[*]}"
    )
    
    # Step 6: merge all hf checkpoints
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py ${REVERSE_MERGE_ARGS[*]}"
    )
    
    # Step 7: cleanup intermediate files
    commands+=(
        "rm -rf ${REVERSE_LANGUAGE_MODEL_PATH}"
        "rm -rf ${REVERSE_VISION_MODEL_PATH}"
        "rm -rf ${REVERSE_ADAPTER_PATH}"
        "rm -rf ${REVERSE_PATCH_PATH}"
    )
else
    echo "当前模型 ${model_name} 不是 VLM 模型，跳过逆向转换"
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
    echo "跳过当前节点的任务【当且仅当只有 是一个master 或者最后一个 worker 节点进行逆向权重转化 !!!】"
fi