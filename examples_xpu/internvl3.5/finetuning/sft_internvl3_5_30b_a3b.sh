source /root/.bashrc
# The script needs to be run on at least 1 nodes.
source activate
conda activate python310_torch25_cuda
#pip uninstall transformer_engine -y
#Clear residual shared memory data
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
ipcrm -m $shmid
echo "Deleted shared memory segment with ID: $shmid"
done

DATA_PATH=${DATA_PATH:-"/mnt/cluster/loongforge/sft_internvl3.5_8b_temp/data-path/webdataset_image"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/loongforge/sft_internvl3.5_30b_a3b_temp/hf-tokenizer-path/InternVL3_5-30B-A3B_20251124144525"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/loongforge/sft_internvl3.5_30b_a3b_temp/load/InternVL3_5-30B-A3B-tp4pp1ep8etp1_20251208111747"}
CHECKPOINT_SAVE_PATH=/mnt/cluster/loongforge/sft_internvl3.5_30b_a3b_temp/save/InternVL3_5-30B-A3B-tp4pp1ep8etp1-save
EXP_NAME=${FULL_JOB_NAME:-"${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"}
LOGS_PATH="${CHECKPOINT_SAVE_PATH}/logs-${EXP_NAME}"
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/rapidfs/users/loongforge/out/tensorboard/internvl3.5/internvl3.5-30b-a3b/stage2-16k-gbs32-tp4pp1ep8-data3-v11/"}
mkdir -p ${CHECKPOINT_SAVE_PATH}
mkdir -p ${LOGS_PATH}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}


####### Use RANK0 to load model and distribute via RDMA #######
export DP_RANK0_LOAD=false

####### Start for NX-P800 #######
export XDNN_USE_FAST_GELU=true  #Added high-precision implementation of the gelu operator
export XMLIR_CUDNN_ENABLED=1  # true indicates the use of cuDNN, which supports conv3d and other functions, while false indicates the non-use of cuDNN
ulimit -c 0
export OMP_NUM_THREADS=1

export NCCL_SOCKET_IFNAME=eth0 # Specify the network interface (such as eth0) used for NCCL communication
export NCCL_IB_HCA=mlx5 # Specify the InfiniBand devices used by NCCL (such as mlx5_0, mlx5_1)
export NCCL_IB_GID_INDEX=3 # Specify the InfiniBand GID index (such as 3) used by NCCL for establishing correct RDMA communication in RoCE networks.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Specify a visible device id
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
export SAVE_LOG_FILE_WITH_RANK_ID=false  # If it is true, the training logs will be stored separately by rank_id

#################################
#export FAST_SWIGLU_ENABLE=1 # Enable swiglu acceleration, refer to SwiGL in python/torch_xmlir/nn/swiglu.py
export XMLIR_ENABLE_FAST_FC=true  # Only used by bf16 in torch.nn.linear.py (LinearWithActFunction, etc.)
#export XMLIR_ENABLE_FAST_FC_FWD_OUT=true #Used for forward output
#export XMLIR_ENABLE_FAST_FC_BWD_DW=true  # Reverse dw is used
export CUDA_DEVICE_ORDER=OAM_ID
export XMLIR_PARALLEL_SAVE_MEMORY=false  # If set to false, the video memory usage will be higher, but there will be a performance improvement; if set to true, the video memory usage will be lower, but performance will decrease
#export XMLIR_DISABLE_CUDA_ALLOCATOR=true
#export USE_FAST_RMS_LAYER_NORM=true
#export XMLIR_DIST_SINGLETON_STREAM=true
#export DIST_MULTI_STREAM=true #Enable multi-stream
export CUDA_DEVICE_MAX_CONNECTIONS=1 # 8 to enable multi-stream
#export XMLIR_BATCH_PARALLEL=true # Used for bf16/fp16; communication fusion operator enabled, USE_CAST_FC_FUSION is auto-disabled under bf16
export XMLIR_DIST_ASYNC_ISEND_IRECV=1 # PP
#export XMLIR_DIST_DISABLE_ASYNC_ISEND_IRECV=1
#export TORCH_C10D_USE_RANDOM_SLEEP=1
export XMLIR_DIST_CHECK_INF_NAN=0  #Check NaN/Inf (default off) 

#################################
#export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 #For 1, allow XDNN to fallback to CPU after an error
#export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #Allow fallback to CPU
#export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true # Generate a list of fallback operators

##############################
export BKCL_RDMA_PROXY_DISABLE=1 # Disable legacy architecture
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_FLAT_RING=1
export BKCL_TREE_THRESHOLD=1  # Set it to 1, which means enabling the tree algorithm
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0   # Set to 1: If there is insufficient space, an OOM error will be reported
export BKCL_ENABLE_XDR=1      # Enable XDR (xpu direct RDMA) and activate direct RDMA. At this point, traffic will flow directly from the XPU to the RDMA network card. This is required for multi-machine training.
export BKCL_RDMA_FORCE_TREE=1 # Enable the tree algorithm
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4  # Based on actual conditions, multiple machines should be configured according to the connectivity of the network cards in the machine environment
export BKCL_RING_BUFFER_SIZE=8388608 # Current optimal setting is 8M
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
# export BKCL_RDMA_VERBS=1  # Used together with BKCL_QPS_PER_CONNECTION; currently only needed for Hygon machines
# export BKCL_QPS_PER_CONNECTION=4  # Current optimal setting: BKCL_QPS_PER_CONNECTION=4

###############################
pkill -9 python || true
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1 # It is recommended to enable xpu_plugin as it brings performance benefits
#################################
######## End for NX-P800 ########

# Change for multinode config
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6024"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
GPUS_PER_NODE=8
# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/internvl3.5/internvl3_5_30b_a3b.yaml

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
    --rotary-seq-len-interpolation-factor 1
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
    --chat-template empty
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length 16384
    --max-position-embeddings 40960
    --max-packed-tokens 16384
    --init-method-std 0.01
    --micro-batch-size 1
    --global-batch-size 32
    --lr 1e-5
    --min-lr 0.0
    --clip-grad 1.0
    --weight-decay 0.05
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --norm-epsilon 1e-6
    --attention-dropout 0
    --hidden-dropout 0
    --train-iters 2000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.03
    --bf16
    --seed 42
    --no-gradient-accumulation-fusion
    #--save-interval 1000
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_SAVE_PATH
    #--dataloader-save ${CHECKPOINT_SAVE_PATH}/dataloader
    --dataloader-type external
    --variable-seq-lengths  # for packing
    --min-num-frame 8
    --max-num-frame 32
    --max-buffer-size 20
    --num-images-expected 48
    --loss-reduction square
    #--use-cpu-initialization
    --use-packed-ds
    # --save-dataset-state
    --use_thumbnail
    --replacement
    --dynamic-image-size
    --loss-reduction-all-gather
    --num-workers 8
    --use-flash-attn
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 42
    --sequence-parallel
    --strict-mode
    --manual-gc
    --manual-gc-interval 0
    --dataloader-prefetch-factor 4
    --exit-interval 500
    --no-bias-dropout-fusion
    --no-bias-gelu-fusion
    --conv-style internvl2_5
    --max-dynamic-patch 12
    --packing-sft-data
    --packing-buffer-size 200
    --energon-pack-algo sequential_max_images
    --allow-missing-adapter-checkpoint
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --context-parallel-size 1
    --use-distributed-optimizer
    --distributed-backend nccl
    --distributed-timeout-minutes 60
)
MOE_ARGS=(
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-2
  #--moe-grouped-gemm
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)
LOGGING_ARGS=(
    --log-interval 1
    #--timing-log-level 2
    #--timing-log-option all
    #--detail-log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
  LOGGING_ARGS+=(
      --wandb-project ${WANDB_PROJECT}
      --wandb-exp-name ${WANDB_NAME}
  )
fi

# Run the training
export PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH"

torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOGGING_ARGS[@]}
