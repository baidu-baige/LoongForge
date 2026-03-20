# Quick Start: VLA Model SFT Training on Kunlunxin P800

## Quick Start: VLA Model SFT Training

This document guides you through the quick start process for fine-tuning Vision-Language-Action (VLA) models using the OmniTraining framework on P800.

Data preparation and weight preparation - TBD (to be supplemented).



## SFT Training Script

OmniTraining currently provides SFT training example scripts for various models. After entering the container, you can find relevant scripts in the `examples_xpu/{model}/finetuning/` directory. Below is an example SFT training script for `PI 0.5`. Please refer to the comments for the purpose of each script section:

```bash
#!/usr/bin/env bash
# Pi05 sanity SFT launcher. This leverages the lightweight pi05 trainer
# (dummy data, single forward/backward) to verify the wiring inside the Omni
# framework. Adjust paths if your repo layout differs.

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/OmniTraining"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}

# Distributed launch (defaults single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
)

# Core training args — pi05 trainer only needs minimal Megatron flags
TRAINING_ARGS=(
    --use-megatron-fsdp
    --training-phase sft
    --micro-batch-size 1
    --global-batch-size 1
    --seq-length 1024
    --max-position-embeddings 1024
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format fsdp_dtensor
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --no-strict-fsdp-dtensor-load
    --finetune
    --seed 1234
    --num-distributed-optimizer-instances 1
    --save-interval 1000000
    --save $CHECKPOINT_PATH
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
)

LOGGING_ARGS=(
    --log-interval 1
)

PYTHONPATH=$MEGATRON_PATH:$BAIGE_OMNI_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $BAIGE_OMNI_PATH/omni_training/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

## Monitoring Logs

By default, the script outputs TensorBoard logs to the directory specified by `TENSORBOARD_PATH`. You can view training curves through TensorBoard.

Additionally, if wandb is installed, you can configure the `WANDB_API_KEY` to upload training metrics to wandb for online monitoring.