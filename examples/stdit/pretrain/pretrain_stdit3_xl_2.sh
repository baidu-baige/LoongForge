#! /bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/aiak-training-llm/dataset/t2v/dog/output.csv_cluster"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/stdit/mcore_stdit3_tp1_pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/tensorborad"}

WANDB_PROJECT=aiak-stdit3-xl/2
WANDB_NAME=$(date '+%Y-%m-%dT%T')
#WANDB_API_KEY=

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

# you can setup stdit maunally
#MODEL_ARGS=(
#    --model-name stdit
#    --num-layers 28
#    --hidden-size 1152
#    --num-attention-heads 16
#    --attention-dropout 0
#    --hidden-dropout 0
#)

# or you can setup stdit by using the following command
MODEL_ARGS=(
    --model-name stdit3-xl/2
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --data-path $DATA_PATH
    --split 949,50,1
)

TRAINING_ARGS=(
    --training-phase pretrain
    --dataloader-type external
    --stdit-bucket-config ${AIAK_TRAINING_PATH}/configs/stdit_bucket_config.json
    --enable-ema
    --ema-decay 0.99
    --latent-frame-interval 1
    --seq-length 0
    --max-position-embeddings 0
    --max-text-length 300
    --init-method-std 0.01
    --micro-batch-size 1
    --global-batch-size 2
    --lr 1e-4
    --min-lr 1e-4
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-15
    --norm-epsilon 1e-6
    --train-iters 10000
    #--lr-decay-iters 6513
    --lr-decay-style constant
    --lr-warmup-iters 1000
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --use-dist-ckpt
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 1000
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --no-save-rng
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --context-parallel-size 4
    --context-parallel-ulysses-degree 4
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
    --timing-log-level 1
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
