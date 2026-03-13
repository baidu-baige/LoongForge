#! /bin/bash
# The script needs to be run on at least 2 nodes.
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Note: Linear attention does not support sequence packing. Therefore, when --packing-sft-data is enabled, 
# we force unpacking of hidden states during the linear-attention stage by setting UNPACKING_HIDDEN_STATES_IN_GDN=1.
# Disabling packing allows linear attention to achieve better computational efficiency and lower memory usage, 
# but it also means that tokens from different sequences may attend to each other.
export UNPACKING_HIDDEN_STATES_IN_GDN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/OmniTraining"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/OmniTraining/dataset/sft_aplaca_zh_data.json"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/OmniTraining/qwen3_next/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/OmniTraining/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/OmniTraining/qwen3_next/Qwen3_Next_80B_A3B_mcore_tp1pp2ep8"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/OmniTraining/tensorboard-log/qwen3-next-80b-a3b"}

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

MODEL_CONFIG_PATH=${AIAK_TRAINING_PATH}/configs/models/qwen3_next/qwen3_next_80b_a3b.yaml

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

MODEL_ARGS=(
    --rotary-base 10000000
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
    # --packing-sft-data
)

SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 32
    --calculate-per-token-loss
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-router-score-function softmax
    --moe-router-topk 10
    --moe-shared-expert-intermediate-size 512
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-overlap
)

TRAINING_ARGS=(
    --seed 42
    --training-phase sft # options: pretrain, sft
    --seq-length 32768
    --max-position-embeddings 32768
    --micro-batch-size 1
    --global-batch-size 64
    --lr 1.0e-5
    --min-lr 1.0e-6
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --attention-softmax-in-fp32
    --clip-grad 1.0
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 1000
    --lr-decay-iters 1000
    --lr-warmup-fraction 0.002
    --lr-decay-style cosine
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 5000000
    --eval-interval 20000000
    --num-workers 32
    --ckpt-format torch
    --no-save-optim
    --no-load-optim
    --no-load-rng

    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 12
    
    --mtp-num-layers 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend flash
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --overlap-grad-reduce
    --overlap-param-gather
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
    $AIAK_TRAINING_PATH/omni_training/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MODEL_CONFIG_ARGS[@]} \
    ${LOGGING_ARGS[@]} 2>&1 | tee /workspace/sft_qwen3_next_80b_a3b.log
