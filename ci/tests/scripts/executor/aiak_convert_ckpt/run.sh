#!/bin/bash
set -x
set -eo pipefail

############################################ 模型训练参数 ############################################

AIAK_MAGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH=${convert_checkpoint_path:-"$AIAK_TRAINING_PATH/tools/convert_checkpoint"}

export PYTHONPATH=$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

CONVERT_ARGS=(
    ${CONVERT_ARGS}
)

final_task=$((WORLD_SIZE - 1))

# 默认命令为空数组
commands=()

# 如果 model_name 是cogvlm2，增加转换脚本，默认会按照顺序执行
if [[ "${model_name}" =~ "cogvlm2" ]]; then
    LANGUAGE_EXPERT_ARGS=(
        ${LANGUAGE_EXPERT_ARGS}
    )
    VISION_EXPERT_ARGS=(
        ${VISION_EXPERT_ARGS}
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

    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/model.py ${LANGUAGE_EXPERT_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/model.py ${VISION_EXPERT_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/model.py ${VISION_MODEL_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/adapter.py  ${ADAPTER_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/vision_patch.py ${PATCH_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/merge_megatron.py ${MERGE_ARGS[*]}"
    )
    mkdir -p $CHECKPOINT_PATH
    echo release > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt
# 如果 model_name 是qwen2-vl系列，增加转换脚本，默认会按照顺序执行
elif [[ "${model_name}" =~ "qwen2-vl" ]]; then
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

    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/model.py ${LANGUAGE_MODEL_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/model.py ${VISION_MODEL_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/adapter.py  ${ADAPTER_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/vision_patch.py ${PATCH_ARGS[*]}"
        "python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/merge_megatron.py ${MERGE_ARGS[*]}"
    )
    mkdir -p $CHECKPOINT_PATH
    echo release > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt
else
    # 将默认命令添加到数组的末尾
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/model.py ${CONVERT_ARGS[*]}"
    )
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