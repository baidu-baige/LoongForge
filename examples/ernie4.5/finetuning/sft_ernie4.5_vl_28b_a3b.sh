#! /bin/bash
# The script needs to be run on at least 1 nodes.
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=/workspace/ernie/Loong-Megatron
LOONGFORGE_PATH=/workspace/ernie/LoongForge/
DATASET_PATH=/workspace/dataset/wds/
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/ernie/ERNIE-4.5-VL-28B-A3B-PT/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/ernie4.5vl/"}
CHECKPOINT_LOAD_PATH=/workspace/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_hg2mcore/
CHECKPOINT_SAVE_PATH=/workspace/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_save/

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
    --position-embedding-type rope
)

MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/ernie4_5_vl/ernie4_5_vl_28b_a3b.yaml

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --seed 42
    --data-path $DATASET_PATH
    --dataloader-type external
)

TRAINING_ARGS=(
    --task-encoder ErnieTaskEncoder
    --packing-sft-data
    --max-packed-tokens 8192
    --max-buffer-size 5
    --packing-buffer-size 20
    --use-fp32-dtype-for-param-pattern "router"
    #--custom-pipeline-layers 4,4,4,4,4,4,2,2
    --training-phase sft
    --chat-template empty
    --seq-length 8192
    --max-position-embeddings 131072
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 4
    --lr 1e-5
    --min-lr 1e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-05
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --no-load-optim
    --no-load-rng
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 20
    --load $CHECKPOINT_LOAD_PATH
    --save $CHECKPOINT_SAVE_PATH
    --save-interval 100
    --exit-interval 100
)

MOE_ARGS=(
    --moe-router-load-balancing-type none
    --moe-router-topk 6
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --empty-unused-memory-level 2
    --moe-router-score-function softmax
    --moe-token-dispatcher-type alltoall
    --moe-router-enable-expert-bias
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    # --sequence-parallel
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 1
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    # --distributed-backend nccl
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


echo "llm path: "  $LOONGFORGE_PATH, "megatron path: " $MEGATRON_PATH
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    +model.image_encoder.freeze=True
