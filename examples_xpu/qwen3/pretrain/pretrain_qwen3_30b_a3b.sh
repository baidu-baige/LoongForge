#! /bin/bash
# The script needs to be run on at least 2 nodes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/OmniTraining"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/OmniTraining/qwen3/pile_test/pile-qwen_text_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen3-30B-A3B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/OmniTraining/qwen3/Qwen3_30B_A3B_mcore_tp2pp2ep4"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/OmniTraining/tensorboard-log/qwen3-30b-a3b"}

GPUS_PER_NODE=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
######################kunlun##########################
# bf16类型专用(megatron相关变量参考<百舸megatron专用>)
export USE_FAST_BF16_FC=true                    # torch.nn.linear.py中用到(LinearWithActFunction等)
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # 默认为false，在特殊情况下(异步checkpoint)需要打开

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # 多机下使用, 以实际情况为准, 多机要按机器环境网卡联通性来配
export BKCL_RDMA_VERBS=1
export BKCL_SOCKET_IFNAME=eth0                  # 以实际情况为准, 默认不开, 找不到网卡时再指定
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # 开1空间不够会报OOM错误
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # 多机alltoall开关， https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/QX0GHLg9-A/fa1a35ef87d947
export BKCL_RING_HOSTID_USE_RANK=1              # 1.2.11版本开始支持, 后续会做为默认的
#export BKCL_FORCE_SYNC=1
#export BKCL_DEBUG=1

export XMLIR_PARALLEL_SAVE_MEMORY=false         # 为false显存占用会多, 但会有性能提升; 为true显存会少, 但性能会下降
export XMLIR_BATCH_PARALLEL=false               # 通信融合算子开启, USE_CAST_FC_FUSION在bf16下会自动失效
export SAVE_LOG_FILE_WITH_RANK_ID=false         # 为true的话, 训练日志会按rank_id分开存储
export XMLIR_LOG_PATH="log-path"                # 指定训练日志的存储目录
export XMLIR_LOG_PREFIX="log-file-prefix"       # 指定训练日志文件名的前缀
export P800_DEBUG=false                         # 为true的话, 训练grad norm出nan会保存ckpt后退出
export P800_DUMP_DIR="ckpt-dump-dir-path"       # 指定训练grad norm出nan会保存ckpt等信息dump的目录
export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # 设为true表示send/recv会走异步逻辑，默认为同步
export XMLIR_CUDNN_ENABLED=1                    # true为使用cuDNN，支持conv3d等，false为不使用cuDNN

# LINEAR 开关
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # 允许某些场景下linear不走xblas fcfusion, 比如走addmm，默认为1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE 走 int32_with_ll, 可选
export XMLIR_MEGATRON_CORE_AIAK_PLUGIN=1

export XMLIR_ENABLE_FAST_FC=true                # 3.3.1.0
export XMLIR_ENABLE_FAST_FC_FWD_OUT=true
export XMLIR_ENABLE_FAST_FC_BWD_DW=true
export XMLIR_ENABLE_FAST_FC_BWD_DX=true

XFLAGS --disable megatron_core_aiak
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
    --model-name qwen3-30b-a3b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
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
    --max-position-embeddings 40960
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
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    #--ckpt-step 0
    #--no-load-optim
    #--no-load-rng
    #--num-workers 8
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    --distributed-backend nccl
    --sequence-parallel
    # --tp-comm-overlap
    # --tp-comm-overlap-bootstrap-backend nccl # or: gloo, mpi
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

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/omni_training/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}