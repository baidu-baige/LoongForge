#! /bin/bash
# The script needs to be run on at least 4 nodes.
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron/"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
DATA_PATH=${DATA_PATH:-"/mnt/cluster/agidriod_validation_1k/metadata.jsonl"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/wan2.1/hg2mcore/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/wan2.1/"}

GPUS_PER_NODE=8

# Change for multi-node config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6008"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --model-name wan2_1_i2v
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --seed 42
    --data-path $DATA_PATH
    --dataloader-type external
)

F=81
H=480
W=832

TRAINING_ARGS=(
    --training-phase pretrain
    --num-latent-frames $F
    --max-latent-height $H
    --max-latent-width $W
    --max-video-length 32760
    --max-text-length 512
    --max-image-length 257
    --max-timestep-boundary 1.0
    --min-timestep-boundary 0.0
    --encoder-seq-length 32760
    --max-position-embeddings 32760
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 16
    --lr 0.0001
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-05
    --norm-epsilon 1e-06
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 5000000
    --no-load-optim
    --no-load-rng
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 20
)

MODEL_PARALLEL_ARGS=(
    --context-parallel-size 8
    --context-parallel-ulysses-degree 8
    --pipeline-model-parallel-size 2

    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
