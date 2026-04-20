#! /bin/bash
# The script needs to be run on at least 2 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json"}

#DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/llama3.1/sft_aplaca_zh_tokenized"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/LoongForge/llama3.1/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/meta-llama/Meta-Llama-3.1-8B/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/llama3.1/mcore_llama3.1_8b_tp1_pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/llama3.1-8b-sft"}

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

# you can setup llama3-8b maunally
#MODEL_ARGS=(
#    --model-name llama3
#    --num-layers 32
#    --hidden-size 4096
#    --ffn-hidden-size 14336
#    --num-attention-heads 32
#    --position-embedding-type rope
#    --normalization RMSNorm
#    --swiglu
#    --attention-dropout 0
#    --hidden-dropout 0
#    --disable-bias-linear
#    --no-position-embedding
#    --untie-embeddings-and-output-weights
#    --group-query-attention
#    --num-query-groups 8
#)

# or you can setup llama3-8b by using the following command
MODEL_ARGS=(
    --model-name llama3.1-8b # options: llama3.1-8b, llama3.1-70b, llama3.1-405b
    --rotary-base 500000
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
)

SFT_ARGS=(
    --chat-template llama3.1
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    #--is-tokenized-data
    #--packing-sft-data
    #--sft-data-streaming

    #--train-on-prompt
    #--eod-mask-loss

    #--sft-dataset-config $DATASET_CONFIG_PATH
    #--sft-dataset sft_aplaca_zh_data # defined in --sft-dataset-config, default: default
    #--data-cache-path $DATA_CACHE_PATH
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length 4096
    --max-position-embeddings 4096
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 128
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-05
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
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
    #--sequence-parallel
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
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
