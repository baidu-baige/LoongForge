#!/bin/bash
set -x
set -eo pipefail

############################################ Preprocess dataset parameters ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
PREPROCESS_DATA_PATH="${PREPROCESS_DATA_PATH}"

PREPROCESS_DATA_ARGS=(
    ${PREPROCESS_DATA_ARGS}
)

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

# Select execution method based on script type
if [[ "${PREPROCESS_DATA_PATH}" == *.sh ]]; then
    # shell script (e.g. offline_packing)
    command="bash $PREPROCESS_DATA_PATH ${PREPROCESS_DATA_ARGS[@]}"
else
    command="python $PREPROCESS_DATA_PATH ${PREPROCESS_DATA_ARGS[@]}"
fi
echo ""
echo "Execute command: $command"
if [[ "${dry_run}" != "true" ]]; then
    eval $command
fi