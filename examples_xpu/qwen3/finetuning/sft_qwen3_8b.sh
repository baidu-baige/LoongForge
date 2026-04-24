#! /bin/bash
# The script needs to be run on at least 2 nodes.

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_tokenized"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen3-8B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen3/Qwen3_8B_mcore_tp1pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-8b-sft"}

GPUS_PER_NODE=8

######################kunlun##########################
# bf16-specific settings (for Megatron-related variables, refer to <Baige Megatron specifics>)
export XMLIR_ENABLE_FAST_FC=true         # Used in torch.nn.linear.py (LinearWithActFunction, etc.)
#export XMLIR_ENABLE_FAST_FC_FWD_OUT=true # Used for forward output
#export XMLIR_ENABLE_FAST_FC_BWD_DW=true  # Used for backward dw
#export XMLIR_ENABLE_FAST_FC_BWD_DX=true  # Used for backward dx
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # Default is false; needs to be enabled in special cases (async checkpoint)

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # Used in multi-node setup, adjust based on actual NIC connectivity
export BKCL_SOCKET_IFNAME=eth0                  # Adjust based on actual environment; disabled by default, specify when NIC is not found
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # Setting to 1 may cause OOM error if space is insufficient
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # Multi-node alltoall switch
export BKCL_RING_HOSTID_USE_RANK=1              # Supported since version 1.2.11, will become the default in the future

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
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1         # Recommended to enable xpu_plugin for performance gains


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
    --model-name qwen3-8b
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
    --is-tokenized-data
    #--packing-sft-data
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
    --max-position-embeddings 32768
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
    --no-load-optim
    --no-load-rng
    #--num-workers 8
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    #--overlap-grad-reduce
    #--overlap-param-gather
    --distributed-backend nccl
    --sequence-parallel
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
