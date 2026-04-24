#! /bin/bash
# The script needs to be run on at least 1 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/llama2/pile_test/pile-llama_text_document"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/meta-llama/Llama-2-7b-hf/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/llama2/mcore_llama2_7b_tp1_pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/llama2-7b-pretrain"}

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

# you can setup llama2-7b maunally
#MODEL_ARGS=(
#    --model-name llama2
#    --num-layers 32
#    --hidden-size 4096
#    --ffn-hidden-size 11008
#    --num-attention-heads 32
#    --position-embedding-type rope
#    --normalization RMSNorm
#    --swiglu
#    --attention-dropout 0
#    --hidden-dropout 0
#    --disable-bias-linear
#    --no-position-embedding
#    --untie-embeddings-and-output-weights
#)

# or you can setup llama2-7b by using the following command
MODEL_ARGS=(
    --model-name llama2-7b # options: llama2-7b, llama2-13b, llama2-70b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 949,50,1
)

TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seq-length 4096
    --max-position-embeddings 4096
    --init-method-std 0.01
    --micro-batch-size 1
    --global-batch-size 1024
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters 500000
    --lr-decay-iters 500000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --fp16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    #--ckpt-step 0
    #--no-load-optim
    #--no-load-rng
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
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
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
