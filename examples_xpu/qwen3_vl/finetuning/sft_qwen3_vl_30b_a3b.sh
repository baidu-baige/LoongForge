#! /bin/bash
# The script needs to be run on at least 2 nodes.
source /root/.bashrc
source activate && conda activate python310_torch25_cuda
 
pkill -9 python || true
 
function check_for_infer() {
    /usr/local/xpu/tools/rw $1 0x300010B8 0
    /usr/local/xpu/tools/rw $1 0x300410B8 0
    /usr/local/xpu/tools/rw $1 0x300810B8 0
    /usr/local/xpu/tools/rw $1 0x310010B8 0
    /usr/local/xpu/tools/rw $1 0x310410B8 0
    /usr/local/xpu/tools/rw $1 0x310810B8 0
    /usr/local/xpu/tools/rw $1 0x320010B8 0
    /usr/local/xpu/tools/rw $1 0x320410B8 0
    /usr/local/xpu/tools/rw $1 0x320810B8 0
    /usr/local/xpu/tools/rw $1 0x330010B8 0
    /usr/local/xpu/tools/rw $1 0x330410B8 0
    /usr/local/xpu/tools/rw $1 0x330810B8 0
}
for ((i=0; i<8; i++))
do
    check_for_infer $i
done
 
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
 
DATA_PATH=${DATA_PATH:-"/mnt/rapidfs/loongforge-training-test/sft_qwen3_vl_30b_a3b_temp/data-path/LLaVA-Pretrain_202511180001/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/rapidfs/loongforge-training-test/sft_qwen3_vl_30b_a3b_temp/hf-tokenizer-path/Qwen3-VL-30B-A3B-Instruct_202512180001/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/rapidfs/loongforge-training-test/sft_qwen3_vl_30b_a3b_temp/load/qwen3-vl-30b-tp4pp1ep8etp1-groupedgemm_202512180001/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/rapidfs/users/loongforge/checkpoints/qwen3-vl/qwen3-vl-30b-tp4pp1ep8etp1-groupedgemm-save/tensorboard-log/"}
 
GPUS_PER_NODE=8
######################kunlun##########################
# bf16-specific settings (for Megatron-related variables, refer to <Baige Megatron specifics>)
export XMLIR_ENABLE_FAST_FC=true                    # Used in torch.nn.linear.py (LinearWithActFunction, etc.)
# export XMLIR_ENABLE_FAST_FC_FWD_OUT=true
# export XMLIR_ENABLE_FAST_FC_BWD_DW=true
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # Default is false; needs to be enabled in special cases (async checkpoint)

export CUDA_DEVICE_MAX_CONNECTIONS=1

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
export SAVE_LOG_FILE_WITH_RANK_ID=false          # If true, training logs will be stored separately by rank_id
export XMLIR_LOG_PATH="/mnt/rapidfs/loongforge-training-test/sft_qwen3_vl_30b_a3b_temp/logs"  # Specify the directory for storing training logs
export XMLIR_LOG_PREFIX="qwen3_vl_30b_sft"      # Specify the prefix for training log filenames
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
    --model-name qwen3-vl-30b-a3b
)
 
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 8
    --chat-template qwen2-vl
    --packing-sft-data
    --packing-buffer-size 1000
    --max-packed-tokens 4096
    --enable-discard-sample
)
 
TRAINING_ARGS=(
    --seed 42
    --norm-epsilon 1e-6
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length 4096
    --max-position-embeddings 262144
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-5
    --min-lr 0.
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --train-iters 100
    --lr-decay-style cosine
    --lr-warmup-fraction 0.03
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_PATH
    --save-interval 10000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
    --no-rope-fusion
    --no-bias-dropout-fusion
    --no-bias-gelu-fusion
    --no-gradient-accumulation-fusion
    --exit-interval 500
)
 
MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    # --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-router-topk 8
    #--empty-unused-memory-level 2
)
 
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    #--overlap-grad-reduce
    #--overlap-param-gather
    --distributed-backend nccl
)
 
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)
 
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    model.image_encoder.apply_rope_fusion=False \
