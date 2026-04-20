#! /bin/bash
# The script needs to be run on at least 4 nodes.
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron/"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATASET_PATH=${DATASET_PATH:-"data/preprocessed"}
#Configure the path to the high noise model and low noise model
HIGH_NOISE_CHECKPOINT_PATH=${HIGH_NOISE_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.2/hg2mcore/high_noise_release/"}
LOW_NOISE_CHECKPOINT_PATH=${LOW_NOISE_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.2/hg2mcore/low_noise_release/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/wan2.2/"}

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
    --model-name wan2-2-i2v
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --seed 42
    --data-path $DATASET_PATH
    --dataloader-type external
)

F=49
H=480
W=832

TRAINING_ARGS=(
    --training-phase pretrain
    --num-latent-frames $F
    --max-latent-height $H
    --max-latent-width $W
    --max-video-length 20280
    --max-text-length 512
    --encoder-seq-length 20280
    --max-position-embeddings 20280
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-5
    --min-lr 1e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-06
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --save-interval 5000000
    --no-load-optim
    --no-load-rng
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 20
)

MODEL_PARALLEL_ARGS=(
    --context-parallel-size 2
    --context-parallel-ulysses-degree 2
    --pipeline-model-parallel-size 4

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

HIGH_NOISE_LOAD_SAVE_ARGS=(
    --load $HIGH_NOISE_CHECKPOINT_PATH
    --save $HIGH_NOISE_CHECKPOINT_PATH
)

HIGH_NOISE_TIMESTEP_BOUNDARY=(
    --max-timestep-boundary 0.358
    --min-timestep-boundary 0.0
)

# Train the high noise model of wan2.2 I2V
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${HIGH_NOISE_LOAD_SAVE_ARGS[@]} \
    ${HIGH_NOISE_TIMESTEP_BOUNDARY[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}

LOW_NOISE_LOAD_SAVE_ARGS=(
    --load $LOW_NOISE_CHECKPOINT_PATH
    --save $LOW_NOISE_CHECKPOINT_PATH
)

LOW_NOISE_TIMESTEP_BOUNDARY=(
    --max-timestep-boundary 1
    --min-timestep-boundary 0.358
)

# Train the low noise model of wan2.2 I2V
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOW_NOISE_LOAD_SAVE_ARGS[@]} \
    ${LOW_NOISE_TIMESTEP_BOUNDARY[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}