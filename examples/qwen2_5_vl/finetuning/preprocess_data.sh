#! /bin/bash
# The script needs to be run on at least 1 nodes.
#export WORLD_SIZE=2
#export RANK=0
#export MASTER_ADDR=192.168.81.53
#export MASTER_PORT=6000

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/aiak-training-llm/dataset/mllm/demo/wds"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/Qwen2.5-VL-7B-Instruct/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/qwen2_5-vl/qwen2_5-vl-7b-tp1-pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/qwen2_5-vl-7b"}

GPUS_PER_NODE=8

# Change for multinode config
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

# or you can setup qwen2_5-vl-7b by using the following command
MODEL_ARGS=(
    --model-name qwen2_5-vl-7b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer \
    --hf-tokenizer-path $TOKENIZER_PATH \
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 8
    --chat-template qwen2-vl
    --packing-batch-size 10
    --packing-sft-data
    --packing-pretrain-data
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules language_model adapter
    --seq-length 4096
    --max-position-embeddings 8192
    --micro-batch-size 1
    --global-batch-size 32
    --train-iters 50000
    --initial-loss-scale 65536
    --bf16
    --save-interval 10000000
    --ckpt-format torch
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size 1
    --tensor-model-parallel-size 1
    --use-distributed-optimizer
    --distributed-backend gloo
    #--sequence-parallel
)

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/tools/data_preprocess/preprocess_megatron_energon_dataset.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    --output-path /mnt/cluster/aiak-training-llm/dataset/mllm/demo/data
