    #! /bin/bash
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
    #################################### Path Configuration ####################################
    MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}      # Megatron-LM 代码路径
    LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}       # LoongForge 训练框架代码路径
    DATA_PATH=${DATA_PATH:-"/mnt/rapidfs/datasets/64k_tokenized_byrepo-11000-13000"}  # 训练数据路径
    TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/rapidfs/models/DeepSeek-V3.1-Terminus-bf16"}  # Tokenizer 路径
    TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/rapidfs/tensorboard-log/deepseek-v31-term-sft"}  # TensorBoard 日志路径
    CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/rapidfs/models/DeepSeek-V3.1-Terminus-tp8pp8ep16etp1"}  # 模型检查点路径

    echo "Using DATA_PATH: ${DATA_PATH}"
    echo "Using TOKENIZER_PATH: ${TOKENIZER_PATH}"
    echo "Using CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
    echo "Using CHECKPOINT_PATH_SAVE: ${CHECKPOINT_PATH_SAVE}"

    ######################kunlun##########################
    # bf16类型专用(megatron相关变量参考<百舸megatron专用>)
    export XMLIR_PARALLEL_SAVE_MEMORY=true         # 为true显存会少, 但性能会下降; 为false显存占用会多, 但会有性能提升
    export XMLIR_ENABLE_FAST_FC_FWD_OUT=true       # 前向输出用到
    export XMLIR_ENABLE_FAST_FC_BWD_DW=true        # 反向 dw 用到
    export XTE_DISABLE_FAST_BF16_CACHE=1           # 禁用 TransformerEngine 的快速 bf16 缓存
    export XTE_DISABLE_MOE_DW_FUSION=0             # MoE 权重梯度融合开关, 0为开启
    export SAVE_LOG_FILE_WITH_RANK_ID=false        # 为true的话, 训练日志会按rank_id分开存储
    export XMLIR_LOG_PATH="log-path"               # 指定训练日志的存储目录
    export XMLIR_LOG_PREFIX="log-file-prefix"      # 指定训练日志文件名的前缀
    export P800_DEBUG=false                        # 为true的话, 训练grad norm出nan会保存ckpt后退出
    export XMLIR_DIST_ASYNC_ISEND_IRECV=false      # 设为true表示send/recv会走异步逻辑，默认为同步
    export XMLIR_CUDNN_ENABLED=1                   # true为使用cuDNN，支持conv3d等，false为不使用cuDNN
    export XMLIR_ENABLE_LINEAR_FC_FUSION=1         # 允许某些场景下linear不走xblas fcfusion, 比如走addmm，默认为1
    export XDNN_FC_GEMM_DTYPE=int32_with_ll        # GEMM_DTYPE 走 int32_with_ll, 可选
    export XMLIR_FUSED_SDP_CHOICE=1                # 0 为 torch._scaled_dot_product_attention 拆算子，1 为 torch._scaled_dot_product_attention 走 xfa
    export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"  # 多机下使用, 以实际情况为准, 多机要按机器环境网卡联通性来配
    export BKCL_SOCKET_IFNAME=eth0                 # 以实际情况为准, 默认不开, 找不到网卡时再指定
    export BKCL_TREE_THRESHOLD=0                   # BKCL 树通信阈值
    export BKCL_FORCE_L3_RDMA=0                    # 开1空间不够会报OOM错误
    export BKCL_ENABLE_XDR=1                       # 启用 XDR 传输
    export BKCL_ALL_TO_ALL_OPT=1                   # 多机alltoall开关
    export BKCL_RING_HOSTID_USE_RANK=1             # 1.2.11版本开始支持, 后续会做为默认的
    export BKCL_RDMA_VERBS=1                       # 与 BKCL_QPS_PER_CONNECTION 配合使用，当前只用于海光机器才需要
    export BKCL_QPS_PER_CONNECTION=4               # 每个连接的 QP 数量
    export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1        # xpu_plugin 推荐开启，有性能收益
    GPUS_PER_NODE=8
    export CUDA_DEVICE_MAX_CONNECTIONS=1           # 开启多流, 提升并发性能
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1       # 避免NCCL记录流, 减少显存占用
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # PyTorch CUDA显存分配配置, 启用可扩展段

    ######################################################
    # Change for multinode config
    MASTER_ADDR=${MASTER_ADDR:-"localhost"}        # 主节点地址, 多机训练时需要修改
    MASTER_PORT=${MASTER_PORT:-"6001"}             # 主节点端口
    NNODES=${WORLD_SIZE:-"16"}                     # 节点数量
    NODE_RANK=${RANK:-"0"}                         # 当前节点rank

    DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    )

    MODEL_ARGS=(
    --model-name deepseek-v3
    --rotary-base 10000
    --original-max-position-embeddings 4096
    --mscale 1.0
    --mscale-all-dim 1.0
    --norm-epsilon 1e-6
    --rotary-scaling-factor 1
    --enable-fa-within-mla
    --use-fp32-dtype-for-param-pattern 'expert_bias'
    )

    DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
    )
    SFT_ARGS=(
    --chat-template no-template
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    --is-tokenized-data
    --packing-sft-data
    --sft-dataset sharegpt
    )

    TRAINING_ARGS=(
    --training-phase sft
    --seq-length 65536
    --use-cpu-initialization
    --max-position-embeddings 163840
    --init-method-std 0.02
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-06
    --train-iters 1500
    --lr-decay-iters 5000
    --lr-decay-style cosine
    --min-lr 1.0e-7
    --weight-decay 0.1
    --lr-warmup-fraction 0.002
    --clip-grad 1.0
    --bf16
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_PATH_SAVE
    --save-interval 1000000
    --eval-interval 30
    --eval-iters 10
    --no-load-optim
    --no-load-rng
    --recompute-granularity full
    --recompute-method block
    --custom-pipeline-layers  8,7,8,8,8,8,8,6
    --custom-pipeline-recompute-layers 8,7,8,8,8,8,8,6
    --reduce-variable-seq-shape-p2p-comm
    --distributed-timeout-minutes 60
    )

    MOE_ARGS=(
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    )

    MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${tp:-8}
    --pipeline-model-parallel-size ${tp:-8}
    --expert-model-parallel-size ${ep:-16}
    --expert-tensor-parallel-size ${etp:-1}
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --use-precision-aware-optimizer
    --manual-gc
    --manual-gc-interval 1
    --use-distributed-optimizer
    --optimizer-cpu-offload
    --optimizer-offload-fraction ${offload_ratio:-"1.0"}
    --overlap-grad-reduce
    --overlap-param-gather
    )

    MTP_ARGS=(
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    )

    LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    )

    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${MTP_ARGS[@]}