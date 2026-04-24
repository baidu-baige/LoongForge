#!/bin/bash
set -x
set -eo pipefail

############################################ mcore to hf checkpoint convert ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
CONVERT_CHECKPOINT_PATH=${convert_checkpoint_path:-"$LOONGFORGE_PATH/tools/convert_checkpoint"}

export LOONGFORGE_PATH
export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

# Parse arguments
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

# VLM Model type detection function
is_vlm_model() {
    local model=$1
    if [[ "${model}" =~ "qwen2.5_vl" ]] || [[ "${model}" =~ "qwen2_vl" ]] || \
       [[ "${model}" =~ "internvl" ]] || [[ "${model}" =~ "llavaov" ]] || \
       [[ "${model}" =~ "qwen3_vl" ]]; then
        return 0
    fi
    return 1
}

# Default commands array is empty
commands=()

# Only VLM models undergo reverse conversion
if is_vlm_model "${model_name}"; then
    # Get parallel parameters
    PP=${PIPELINE_MODEL_PARALLER_SIZE:-2}
    ETP=${ENCODER_TENSOR_MODEL_PARALLER_SIZE:-2}
    DTP=${DECODER_TENSOR_MODEL_PARALLER_SIZE:-2}
    
    # Select adapter conversion script based on model type
    if [[ "${model_name}" =~ "internvl" ]]; then
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py"
    else
        ADAPTER_SCRIPT="$CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py"
    fi

    # Step 1: key reverser - Convert omni format mcore weights to standard mcore format
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py ${REVERSE_KEY_ARGS[*]}"
    )
    
    # Step 2: language model mcore -> hf
    commands+=(
        "python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py ${REVERSE_LANGUAGE_MODEL_ARGS[*]}"
    )
    
    # Step 3: vision model mcore -> hf
    # Handle PP > 1 case, need to create temporary directory
    if [[ $PP -gt 1 ]]; then
        # Allow overriding load path via environment variable, defaults to release
        MCORE_LOAD_DIR=${MCORE_LOAD_PATH:-"${CHECKPOINT_PATH}/release"}
        LOAD_PATH=${MCORE_LOAD_DIR}/tmp/

        commands+=(
            "mkdir -p $LOAD_PATH"
        )
        # Copy vision model related files
        for ((i=0;i<$ETP;i++)); do
            from=$(printf "mp_rank_%02d_000" $i)
            to=$(printf "mp_rank_%02d" $i)
            commands+=(
                "cp -r ${MCORE_LOAD_DIR}/$from $LOAD_PATH/$to"
            )
        done
        # Update load_ckpt_path in vision model arguments
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
    echo "Current model ${model_name} is not a VLM model, skipping reverse conversion"
fi


# Determine if it is the only master or the last worker running
if [[ "${RANK}" != "" ]] && [[ "${RANK}" == "${final_task}" ]] && [[ "${dry_run}" != "true" ]]; then
    echo ""
    # Iterate through command array and execute each command
    for command in "${commands[@]}"; do
        echo "Execute command: \"$command\""
        eval "$command"
    done
else
    echo "Skip current node task [Only if it is a master or last worker node to perform reverse weight conversion !!!]"
fi