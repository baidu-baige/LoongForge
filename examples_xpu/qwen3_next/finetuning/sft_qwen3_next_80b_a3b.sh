#! /bin/bash
# The script needs to be run on at least 2 nodes.

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/users/dataset/qwen3-next/sft_aplaca_zh_new_data.json"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/users/dataset/qwen3-next/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

# Common paths and configurations
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/users/checkpoints/qwen3-next/Qwen3-Next-80B-A3B-Instruct/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/users/checkpoints/qwen3-next/Qwen3-Next-80B-A3B-tp2pp4ep8etp1-mtp/"}
CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH:-"/mnt/cluster/users/checkpoints/qwen3-next/Qwen3-Next-80B-A3B-tp2pp4ep8etp1-mtp-save/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/users/out/tensorboard/qwen3-next"}

export XMLIR_MOCK_ASYNC_LINEAR=0 

GPUS_PER_NODE=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
######################kunlun##########################
# bf16-specific settings (for Megatron-related variables, refer to <Baige Megatron specifics>)
export XDNN_USE_FAST_GELU=false
export XMLIR_ENABLE_FAST_FC=true                    # Used in torch.nn.linear.py (LinearWithActFunction, etc.)
# export XMLIR_ENABLE_FAST_FC_FWD_OUT=true
# export XMLIR_ENABLE_FAST_FC_BWD_DW=true
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # Default is false; needs to be enabled in special cases (async checkpoint)

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # Used in multi-node setup, adjust based on actual NIC connectivity
export BKCL_SOCKET_IFNAME=eth0                  # Adjust based on actual environment; disabled by default, specify when NIC is not found
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # Setting to 1 may cause OOM error if space is insufficient
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # Multi-node alltoall switch
export BKCL_RING_HOSTID_USE_RANK=1              # Supported since version 1.2.11, will become the default in the future
export BKCL_RDMA_VERBS=1

export XMLIR_PARALLEL_SAVE_MEMORY=false         # false: higher memory usage but better performance; true: lower memory usage but reduced performance
export XMLIR_BATCH_PARALLEL=false               # Communication fusion operator enabled; USE_CAST_FC_FUSION is auto-disabled under bf16
export SAVE_LOG_FILE_WITH_RANK_ID=false         # If true, training logs will be stored separately by rank_id
export XMLIR_LOG_PATH="log-path"                # Specify the directory for storing training logs
export XMLIR_LOG_PREFIX="log-file-prefix"       # Specify the prefix for training log filenames
export P800_DEBUG=false                         # If true, will save checkpoint and exit when grad norm is NaN
export P800_DUMP_DIR="ckpt-dump-dir-path"       # Specify the directory to dump checkpoint and other info when grad norm is NaN
export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # Set to true for async send/recv logic; default is synchronous
export XMLIR_CUDNN_ENABLED=1                    # true to use cuDNN (supports conv3d, etc.); false to disable cuDNN

# LINEAR switch
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # Allow linear to bypass xblas fcfusion in certain scenarios, e.g., use addmm; default is 1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE uses int32_with_ll, optional
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1

XFLAGS --disable transformer_engine_1_7
XFLAGS --disable transformer_engine_1_13
######################################################
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
    --model-name qwen3-next-80b-a3b
    --rotary-base 10000000
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
    #--packing-sft-data
)
#export UNPACKING_HIDDEN_STATES_IN_GDN=1
SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 32
    --calculate-per-token-loss
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    #--moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-router-score-function softmax
    --moe-router-topk 10
    --moe-shared-expert-intermediate-size 512
    --moe-token-dispatcher-type alltoall
    #--moe-shared-expert-overlap
)

TRAINING_ARGS=(
    --seed 42
    --training-phase sft # options: pretrain, sft
    --seq-length 32768
    --max-position-embeddings 32768
    --micro-batch-size 1
    --global-batch-size 32
    --lr 1.0e-5
    --min-lr 0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --attention-softmax-in-fp32
    --clip-grad 1.0
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 2000
    --lr-decay-iters 2000
    --lr-warmup-fraction 0.03
    --lr-decay-style cosine
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_SAVE_PATH
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
    --exit-interval 200
    --mtp-num-layers 1
    #--mtp-loss-scaling-factor 0
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend flash
    #--cross-entropy-loss-fusion
    #--cross-entropy-fusion-impl native
#    --overlap-grad-reduce
#    --overlap-param-gather
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
