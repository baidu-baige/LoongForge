#! /bin/bash
# The script needs to be run on at least 1 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=5

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/workspace/loongforge-ckpt/pile_test/pile-deepseek_text_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/loongforge-ckpt/Qwen3-1.7B-Instruct"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/loongforge-ckpt/Qwen3-1.7B-Instruct"}
# CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/loongforge-ckpt/qwen3_1.7B_mcore_tp1pp1"}

LORA_CHECKPOINT_PATH=${LORA_CHECKPOINT_PATH:-"/workspace/loongforge-ckpt/qwen3_1.7B_mcore_tp1pp1_lora"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-1.7b"}


GPUS_PER_NODE=1

# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/qwen3/qwen3_1_7b_lora.yaml

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12435"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

MODEL_ARGS=(
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
    --attention-backend fused
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --eod-mask-loss
    --data-path $DATA_PATH
    --split 99,1,0
)

TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seq-length 4096
    --max-position-embeddings 32768
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 2
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 50
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $LORA_CHECKPOINT_PATH
    --save $LORA_CHECKPOINT_PATH
    --pretrained-checkpoint $CHECKPOINT_PATH
    --save-interval 40 
    --save-hf true
    --save-hf-path /workspace/loongforge-ckpt/qwen3-1.7b-lora-hf
    --lora-alpha 32
    --lora-dim 16
    --eval-interval 1000
    --eval-iters 10
    #--ckpt-step 0
    #--no-load-optim
    #--no-load-rng
    #--num-workers 8
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
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
