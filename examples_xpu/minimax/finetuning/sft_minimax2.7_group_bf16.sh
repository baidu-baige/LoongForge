#!/bin/bash
# This script is used for SFT training Minimax2.x in BF16 mixed precision.
# The script needs to be run on at least 16 nodes for 128k seqlength.
source activate && conda activate python310_torch25_cuda
pkill -9 python || true

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

#DATA_PATH=${DATA_PATH:-"/mnt/rapidfs/datasets/tigerbot-alpaca-zh-0.5m.json"}
DATA_PATH="/mnt/rapidfs/datasets/128k-packing-minimax-tigerbot-alpaca-zh"
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/rapidfs/models/MiniMax-M2.7-BF16/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/rapidfs/LoongForge/mini_max/MiniMax_m2_7_mcore_tp8pp16ep8etp1/"}
CHECKPOINT_SAVE_PATH="/mnt/rapidfs/LoongForge/ckpt/minimax_m2_7/allmodel"
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/rapidfs/LoongForge/tensorboard-log/minimax_m2_7/allmodel"}

GPUS_PER_NODE=8

export XMLIR_ENABLE_FAST_FC=true              # Used in torch.nn.linear.py (LinearWithActFunction etc.)
export XMLIR_ENABLE_FAST_FC_FWD_OUT=true      # Used for forward output
#export XMLIR_ENABLE_FAST_FC_BWD_DW=true      # Used for backward dw
#export XMLIR_ENABLE_FAST_FC_BWD_DX=true      # Used for backward dx
export XTE_DISABLE_FAST_BF16_CACHE=1          # Disable cache for mixed precision fwd
export XTE_DISABLE_MOE_DW_FUSION=0            # MoE DW fusion (critical)
export XMLIR_FUSED_SDP_CHOICE=1               # XFA flash attention

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # For multi-node, adjust based on actual NIC connectivity
export BKCL_RDMA_VERBS=1
export BKCL_SOCKET_IFNAME=eth0                  # Adjust based on actual setup; disabled by default, specify when NIC not found
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # Setting to 1 may cause OOM if space is insufficient
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # Multi-node alltoall switch
export BKCL_RING_HOSTID_USE_RANK=1              # Supported since v1.2.11, will be default in future

export XMLIR_PARALLEL_SAVE_MEMORY=false         # false: more memory but better performance; true: less memory but worse performance
export XMLIR_BATCH_PARALLEL=false               # Enable communication fusion; USE_CAST_FC_FUSION is automatically disabled under bf16
export SAVE_LOG_FILE_WITH_RANK_ID=false         # If true, training logs are stored separately by rank_id
export XMLIR_LOG_PATH="log-path"                # Specify training log directory
export XMLIR_LOG_PREFIX="log-file-prefix"       # Specify training log filename prefix
export P800_DEBUG=false                         # If true, saves checkpoint and exits when grad norm is NaN
export P800_DUMP_DIR="ckpt-dump-dir-path"       # Specify dump directory for checkpoint/info when grad norm is NaN

export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # If true, send/recv uses async logic; default is sync
export XMLIR_CUDNN_ENABLED=1                    # true: use cuDNN (supports conv3d etc.); false: disable cuDNN
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # Allow linear to skip xblas fcfusion in some cases (e.g., use addmm); default is 1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE uses int32_with_ll; optional
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1    

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=6657
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)


SFT_ARGS=(
  --chat-template minimax-m2
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
  --is-tokenized-data
  #--use-fixed-seq-lengths
)

MODEL_ARGS=(
  --model-name minimax2.7-230b
  --rotary-percent 0.5
  --norm-epsilon 1e-6
  --rotary-base 5000000
  --use-fp32-dtype-for-param-pattern expert_bias
  --attention-backend flash
  
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 131072
  --max-position-embeddings 131072  # not used
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 32
  --lr 1e-6
  --train-iters 1000
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_SAVE_PATH
  --save-interval 200
  --eval-interval 10000
  --eval-iters 10000
  --exit-interval 200
  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  #--custom-pipeline-layers 16,16,16,14
  #--custom-pipeline-recompute-layers 16,16,16,14
  --custom-pipeline-layers 3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3
  --custom-pipeline-recompute-layers 3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  #--manual-gc
  #--manual-gc-interval 1
  #--cross-entropy-loss-fusion
)

MOE_ARGS=(
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-score-function sigmoid
  --moe-router-dtype fp32
  #--empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 16
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type alltoall
  #--moe-enable-deepep
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  #--moe-permute-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)


LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  #--check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${SFT_ARGS[@]} 