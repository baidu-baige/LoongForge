#!/bin/bash
set -x
set -eo pipefail

############################################ Preprocess dataset parameters ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
PREPROCESS_DATA_PATH="${PREPROCESS_DATA_PATH}"

PREPROCESS_DATA_ARGS=(
    ${PREPROCESS_DATA_ARGS}
)

export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

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