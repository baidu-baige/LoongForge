#!/bin/bash
set -x
set -eo pipefail

############################################ Model training parameters ############################################

AIAK_MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/ssd1/workspace/AIAK-Training-Omni"}
# AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH=${convert_checkpoint_path:-"$AIAK_TRAINING_PATH/tools/convert_checkpoint"}

export PYTHONPATH=$AIAK_MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

CONVERT_ARGS=(
    ${CONVERT_ARGS}
)

final_task=$((WORLD_SIZE - 1))

# Default command is empty array
commands=()

# VLM model type detection function
is_vlm_model() {
    local model=$1
    if [[ "${model}" =~ "qwen2.5_vl" ]] || [[ "${model}" =~ "qwen2_vl" ]] || \
       [[ "${model}" =~ "internvl" ]] || [[ "${model}" =~ "llavaov" ]] || \
       [[ "${model}" =~ "qwen3_vl" ]]; then
        return 0
    fi
    return 1
}

# MoE model type detection function
is_moe_model() {
    local model=$1
    if [[ "${model}" =~ "a3b" ]] || [[ "${model}" =~ "moe" ]] || \
       [[ "${model}" =~ "deepseek_v2" ]] || [[ "${model}" =~ "deepseek_v3" ]]; then
        return 0
    fi
    return 1
}

# If VLM model (qwen2.5_vl, internvl, llavaov, etc.), use new module_convertor approach
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

    # Select adapter conversion script based on model type
    if [[ "${model_name}" =~ "internvl" ]]; then
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py"
    else
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py"
    fi

    # Select merge script based on model type (MoE models use merge_megatron_expert.py)
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
    # LLM models use default conversion method
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${CONVERT_ARGS[*]}"
    )
    mkdir -p $CHECKPOINT_PATH
    echo release > $CHECKPOINT_PATH/latest_checkpointed_iteration.txt
fi


# Only run on master or the last worker
if [[ "${RANK}" != "" ]] && [[ "${RANK}" == "${final_task}" ]] && [[ "${dry_run}" != "true" ]]; then
    echo ""
    # Iterate through command array and execute each command
    for command in "${commands[@]}"; do
        echo "Executing command: \"$command\""
        eval "$command"
    done
else
    echo "Skipping task on current node [Only when master or the last worker node performs weight conversion !!!]"
fi