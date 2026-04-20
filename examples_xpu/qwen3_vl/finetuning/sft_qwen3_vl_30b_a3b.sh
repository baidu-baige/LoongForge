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
# bf16类型专用(megatron相关变量参考<百舸megatron专用>)
export XMLIR_ENABLE_FAST_FC=true                    # torch.nn.linear.py中用到(LinearWithActFunction等)
# export XMLIR_ENABLE_FAST_FC_FWD_OUT=true
# export XMLIR_ENABLE_FAST_FC_BWD_DW=true
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # 默认为false，在特殊情况下(异步checkpoint)需要打开
 
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # 多机下使用, 以实际情况为准, 多机要按机器环境网卡联通性来配
export BKCL_SOCKET_IFNAME=eth0                  # 以实际情况为准, 默认不开, 找不到网卡时再指定
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # 开1空间不够会报OOM错误
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # 多机alltoall开关， https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/QX0GHLg9-A/fa1a35ef87d947
export BKCL_RING_HOSTID_USE_RANK=1              # 1.2.11版本开始支持, 后续会做为默认的
export BKCL_RDMA_VERBS=1
 
export XMLIR_PARALLEL_SAVE_MEMORY=false         # 为false显存占用会多, 但会有性能提升; 为true显存会少, 但性能会下降
export XMLIR_BATCH_PARALLEL=false               # 通信融合算子开启, USE_CAST_FC_FUSION在bf16下会自动失效
export SAVE_LOG_FILE_WITH_RANK_ID=false          # 为true的话, 训练日志会按rank_id分开存储
export XMLIR_LOG_PATH="/mnt/rapidfs/loongforge-training-test/sft_qwen3_vl_30b_a3b_temp/logs"  # 指定训练日志的存储目录
export XMLIR_LOG_PREFIX="qwen3_vl_30b_sft"      # 指定训练日志文件名的前缀
export P800_DEBUG=false                         # 为true的话, 训练grad norm出nan会保存ckpt后退出
export P800_DUMP_DIR="ckpt-dump-dir-path"       # 指定训练grad norm出nan会保存ckpt等信息dump的目录
export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # 设为true表示send/recv会走异步逻辑，默认为同步
export XMLIR_CUDNN_ENABLED=1                    # true为使用cuDNN，支持conv3d等，false为不使用cuDNN
 
# LINEAR 开关
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # 允许某些场景下linear不走xblas fcfusion, 比如走addmm，默认为1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE 走 int32_with_ll, 可选
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
