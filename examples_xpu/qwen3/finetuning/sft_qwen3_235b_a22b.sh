#! /bin/bash
# The script needs to be run on at least 4 nodes.
#source activate && conda activate python310_torch25_cuda
set -x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/datasets/qwen3/tigerbot-alpaca-zh-0.5m_tokenized"}
DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen3-235B-A22B/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/Qwen3_235B_A22B_mcore_tp4pp4ep8etp1"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-235b-a22b"}

GPUS_PER_NODE=8

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0

export XMLIR_ENABLE_FAST_FC=true        # Used in torch.nn.linear.py (LinearWithActFunction, etc.)
#export XMLIR_ENABLE_FAST_FC_FWD_OUT=true # Used for forward output
#export XMLIR_ENABLE_FAST_FC_BWD_DW=true  # Used for backward dw
#export XMLIR_ENABLE_FAST_FC_BWD_DX=true  # Used for backward dx
#### P800 environment start ####
# bf16-specific settings (for Megatron-related variables, refer to <Baige Megatron specifics>)
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
export P800_DEBUG=false                         # If true, will save checkpoint and exit when grad norm is NaN
#export P800_DUMP_DIR="ckpt-dump-dir-path"       # Specify the directory to dump checkpoint and other info when grad norm is NaN
export XMLIR_DIST_ASYNC_ISEND_IRECV=false        # Set to true for async send/recv logic; default is synchronous
export XMLIR_CUDNN_ENABLED=1                    # true to use cuDNN (supports conv3d, etc.); false to disable cuDNN
export XDNN_FC_GEMM_DTYPE=int32_with_ll
# LINEAR switch
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # Allow linear to bypass xblas fcfusion in certain scenarios, e.g., use addmm instead
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1         # Recommended to enable xpu_plugin for performance gains

#XFLAGS --disable transformer_engine_1_7
XFLAGS --disable transformer_engine_1_13
#### P800 environment end #### 
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"4"}
NODE_RANK=${RANK:-"0"}
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)
MODEL_ARGS=(
    --model-name qwen3-235b-a22b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --eod-mask-loss
    --data-path $DATA_PATH
    --split 100,0,0
)
SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    --is-tokenized-data
    --packing-sft-data
)
TRAINING_ARGS=(
    --training-phase sft # options: pretrain, sft
    --seq-length 4096
    --max-position-embeddings 131072
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
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    --recompute-granularity full
    --recompute-method block
    --custom-pipeline-layers 23,23,24,24
    #--custom-pipeline-recompute-layers 23,23,24,24
    --recompute-num-layers 12
)
MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    #--moe-router-force-load-balancing
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --distributed-backend nccl
    --sequence-parallel
    --optimizer-cpu-offload
    --use-precision-aware-optimizer
    --optimizer-offload-fraction 1.0
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