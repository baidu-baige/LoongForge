#! /bin/bash
# The script needs to be run on at least 2 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/mllm/demo/wds/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen3-VL-30B-A3B-Instruct"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen3-vl/qwen3-vl-30b-tp2pp2ep4"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-vl-30b-a3b"}

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

# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --add-question-in-pretrain
    --enable-discard-sample
    --num-workers 16
    # --packing-sft-data
    # --packing-buffer-size 500
)

TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seq-length 32768
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 32
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    #--load $CHECKPOINT_PATH
    #--save $CHECKPOINT_PATH
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    # --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
    # --tp-comm-overlap
    # --tp-comm-overlap-bootstrap-backend nccl # or: gloo, mpi
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
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
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}