#! /bin/bash
# The script needs to be run on at least 2 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/tigerbot-alpaca-zh-0.5m.json"}

# DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_tokenized"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen3-30B-A3B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen3/Qwen3_30B_A3B_mcore_tp2pp2ep4"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-30b-a3b"}

GPUS_PER_NODE=8

export CUDA_DEVICE_MAX_CONNECTIONS=1

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

MODEL_ARGS=(
    --model-name qwen3-30b-a3b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
)

SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    #--is-tokenized-data
    --packing-sft-data
    #--sft-data-streaming

    #--train-on-prompt
    #--eod-mask-loss

    #--sft-dataset-config $DATASET_CONFIG_PATH
    #--sft-dataset sft_aplaca_zh_data # defined in --sft-dataset-config, default: default
    #--data-cache-path $DATA_CACHE_PATH
)

TRAINING_ARGS=(
    --training-phase sft # options: pretrain, sft
    --seq-length 4096
    --max-position-embeddings 40960
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 5000
    --lr-decay-iters 5000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    #--ckpt-step 0
    #--no-load-optim
    #--no-load-rng
    #--num-workers 8
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    --distributed-backend nccl
    --sequence-parallel
    # --tp-comm-overlap
    # --tp-comm-overlap-bootstrap-backend nccl # or: gloo, mpi
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

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
