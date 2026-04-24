#! /bin/bash
# The script needs to be run on at least 1 nodes.
source /root/.bashrc
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
ipcrm -m $shmid
echo "Deleted shared memory segment with ID: $shmid"
done

DATA_PATH=/mnt/cluster/LoongForge/dataset/internvl/webdataset
TOKENIZER_PATH=/mnt/cluster/huggingface.co/internvl/InternVL2_5-8B/
CHECKPOINT_LOAD_PATH=/mnt/cluster/LoongForge/internvl2.5/internvl2.5-8b-tp4-pp1-Nov24
CHECKPOINT_SAVE_PATH=/mnt/cluster/LoongForge/internvl2.5/internvl2.5-8b-tp4-pp1-save
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/out/tensorboard/internvl2.5/internvl2.5-8b/"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
GPUS_PER_NODE=8

# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/internvl2.5/internvl2_5_8b.yaml

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
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
  --norm-epsilon 1e-6
  --attention-dropout 0
  --hidden-dropout 0
  --train-iters 2000
  --lr-decay-style cosine
  --lr-warmup-fraction 0.03
  --bf16
  --seed 42
  --no-gradient-accumulation-fusion
  --load $CHECKPOINT_LOAD_PATH
  --save $CHECKPOINT_SAVE_PATH
  --save-interval 2000
  --exit-interval 500
  --dataloader-type external
  --variable-seq-lengths  
  --min-num-frame 8
  --max-num-frame 32
  --max-buffer-size 20
  --num-images-expected 48
  --loss-reduction square
  #--use-cpu-initialization
  #--no-initialization
  --use_thumbnail
  --replacement
  --dynamic-image-size
  --loss-reduction-all-gather
  --num-workers 8
  --dataloader-prefetch-factor 4
  --use-flash-attn
  --recompute-granularity full
  --recompute-method block
  --recompute-num-layers 12
  --sequence-parallel
  --strict-mode
  --conv-style internvl2_5
  --max-dynamic-patch 12
  --packing-sft-data
  --packing-buffer-size 20
  --energon-pack-algo sequential_max_images
  --allow-missing-adapter-checkpoint
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 1
  --use-distributed-optimizer
  --context-parallel-size 1
  --distributed-backend nccl
)

MODEL_CONFIG_ARGS=(
  --config-file $MODEL_CONFIG_PATH
  --rope-scaling-factor 2
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
  ${MODEL_CONFIG_ARGS[@]} \
  --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  #+model.image_encoder.freeze=True \
  #+model.image_projector.freeze=True \
  #+model.foundation.freeze=True