#! /bin/bash
# The script needs to be run on at least 1 nodes.
source /root/.bashrc

#清除残留的共享内存数据
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
ipcrm -m $shmid
echo "Deleted shared memory segment with ID: $shmid"
done

DATA_ROOT=${DATA_ROOT:-"/mnt/cluster/hejinhui/data/qianfan/filter_CC3M#/mnt/cluster/hejinhui/data/qianfan/filter_mmdu"}
DATA_ANNOTATIONS=${DATA_ANNOTATIONS:-"/mnt/cluster/hejinhui/data/qianfan/filter_CC3M/LLaVA-ReCap-CC3M-formatted_correctandsampled0207_ref.jsonl#/mnt/cluster/hejinhui/data/qianfan/filter_mmdu/mmdu-45k_formatted_filterd_0202.jsonl"}
DATA_AUGMENT=${DATA_AUGMENT:-"false#false"}
DATA_MAX_DYNAMIC_PATCH=${DATA_MAX_DYNAMIC_PATCH:-"12#12"}
DATA_REPEAT_TIME=${DATA_REPEAT_TIME:-"1#1"}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

DATA_PATH=$(join_by @ "$DATA_ROOT" "$DATA_ANNOTATIONS" "$DATA_AUGMENT" "$DATA_MAX_DYNAMIC_PATCH" "$DATA_REPEAT_TIME")
echo $DATA_PATH

TOKENIZER_PATH=/mnt/cluster/zhaiyanfeng/models/internvl/ckpt/InternVL2_5-26B/
CHECKPOINT_LOAD_PATH=/mnt/cluster/zhaiyanfeng/models/internvl/ckpt-megatron/internvl2.5-26b-tp4-pp1
CHECKPOINT_SAVE_PATH=/mnt/cluster/zhaiyanfeng/models/internvl/ckpt-megatron/internvl2.5-26b-tp4-pp1-save
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/zhaiyanfeng/out/tensorboard/internvl2.5-master/26b-stage2-16k-1node-gbs32/"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron/"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
GPUS_PER_NODE=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MODEL_ARGS=(
  --model-name internvl2.5-26b
  --rotary-base 50000000  # for internvl
  --rope-scaling-factor 2.5
)


DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  #--additional-special-tokens "<img>,</img>,<IMG_CONTEXT>,<quad>,</quad>,<ref>,</ref>,<box>,</box>"
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 16384
  --max-position-embeddings 32768
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
  --norm-epsilon 1e-5
  --attention-dropout 0
  --hidden-dropout 0
  --train-iters 2000
  --lr-decay-style cosine
  --lr-warmup-fraction 0.03
  --bf16
  --trainable-modules adapter vision_model language_model # stage 2
  --seed 42
  --no-gradient-accumulation-fusion
  --load $CHECKPOINT_LOAD_PATH
  --save $CHECKPOINT_SAVE_PATH
  --save-interval 2000
  --exit-interval 500
  --dataloader-type external
  --variable-seq-lengths  # for packing
  --min-num-frame 8
  --max-num-frame 32
  --max-buffer-size 20
  --num-images-expected 48
  --loss-reduction square
  #--use-cpu-initialization
  --no-initialization
  --use-packed-ds
  --use_thumbnail
  --replacement
  --dynamic-image-size
  --loss-reduction-all-gather
  --num-workers 8
  --dataloader-prefetch-factor 4
  --use-flash-attn
  --recompute-granularity full
  --recompute-method block
  --recompute-num-layers 48
  --sequence-parallel
  --strict-mode
  --conv-style internvl2_5
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 1
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --use-precision-aware-optimizer
  --overlap-cpu-optimizer-d2h-h2d
  #--custom-pipeline-layers 8,40
  #--custom-pipeline-recompute-layers 42,46
  --use-distributed-optimizer
  --context-parallel-size 1
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

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]}