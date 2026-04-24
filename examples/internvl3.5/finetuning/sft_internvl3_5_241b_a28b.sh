#! /bin/bash
# The script needs to be run on at least 4 nodes.
source /root/.bashrc

# Clear remaining shared memory data
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
ipcrm -m $shmid
echo "Deleted shared memory segment with ID: $shmid"
done

DATA_PATH=${DATA_PATH:-"/workspace/dataset/filter_mmdu/webdataset/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/InternVL3_5-241B-A28B/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/data/checkpoint/InternVL3_5-241B-A28B-tp4pp4ep8etp1-without-gemm"}
CHECKPOINT_SAVE_PATH=/mnt/data/checkpoint/InternVL3_5-241B-A28B-tp4pp4ep8etp1-11-28-28-27-save
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/data/zhaiyanfeng/out/tensorboard/internvl3.5/internvl3.5-241b-a28b/stage2-8k-gbs32-tp4pp4ep8-4nodes/"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron/"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
GPUS_PER_NODE=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/internvl3.5/internvl3_5_241b_a28b.yaml

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MODEL_ARGS=(
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
  --seq-length 8192
  --max-position-embeddings 40960
  --max-packed-tokens 8192
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
  --load $CHECKPOINT_PATH
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
  --use_thumbnail
  --replacement
  --dynamic-image-size
  --loss-reduction-all-gather
  --num-workers 8
  --dataloader-prefetch-factor 4
  --manual-gc
  --manual-gc-interval 0
  --use-flash-attn
  --recompute-granularity full
  --recompute-method block
  --recompute-num-layers 45
  --sequence-parallel
  --strict-mode
  --conv-style internvl2_5
  --no-bias-dropout-fusion
  --no-bias-gelu-fusion
  --max-dynamic-patch 12
  --packing-sft-data
  --packing-buffer-size 200
  --energon-pack-algo sequential_max_images
  --allow-missing-adapter-checkpoint
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    #--moe-grouped-gemm
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 4
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --use-precision-aware-optimizer
  --overlap-cpu-optimizer-d2h-h2d
  --custom-pipeline-layers 11,28,28,27
  --use-distributed-optimizer
  --context-parallel-size 1
  --distributed-backend nccl
)

LOGGING_ARGS=(
  --log-interval 1
  #--timing-log-level 2
  #--timing-log-option all
  #--detail-log-interval 3
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
  --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml \
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]}
