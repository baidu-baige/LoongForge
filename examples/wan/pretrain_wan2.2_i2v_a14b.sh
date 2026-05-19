#! /bin/bash
# WAN2.2 I2V training with Megatron FSDP ZeRO-3.

# CUDA_DEVICE_MAX_CONNECTIONS must be UNSET for FSDP (not set to 1)
unset CUDA_DEVICE_MAX_CONNECTIONS
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/home/opt/cuda_tools/:$PATH
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/wan2.2/Loong-Megatron/"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/wan2.2/LoongForge/"}
DATASET_PATH=${DATASET_PATH:-"/workspace/wan2.2/LoongForge/examples/wan/data/preprocessed/"}
# High-noise and low-noise checkpoint paths.
HIGH_NOISE_CHECKPOINT_PATH=${HIGH_NOISE_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.2/hg2mcore/high_noise/Megatron_Release/"}
LOW_NOISE_CHECKPOINT_PATH=${LOW_NOISE_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.2/hg2mcore/low_noise/Megatron_Release/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/wan2.2/"}

GPUS_PER_NODE=${GPUS:-8}

# Distributed launch configuration.
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6007"}
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
    --model-name wan2-2-i2v
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --seed 42
    --data-path $DATASET_PATH
    --dataloader-type external
)

F=49
#F=205
H=480
W=832
F_POST=$((($F - 1) / 4 + 1))           # VAE temporal compress (WAN 2.2 = 4)
H_POST=$(($H / 8 / 2))                  # VAE spatial(8) + patch_h(2)
W_POST=$(($W / 8 / 2))                  # VAE spatial(8) + patch_w(2)
SEQ_LEN=$(($F_POST * $H_POST * $W_POST))

# Activation recompute. Wan2.2 has 40 DiT layers; block mode recomputes the first N layers.
# Default N=32 leaves the last 8 layers without recompute for better throughput while keeping memory safe.
RECOMPUTE_MODE=${RECOMPUTE_MODE:-"full"}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-"32"}

if [ "${RECOMPUTE_MODE}" = "selective" ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity selective
    )
elif [ "${RECOMPUTE_MODE}" = "none" ]; then
    RECOMPUTE_ARGS=()
else
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method block
        --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}
    )
fi

TRAINING_ARGS=(
    --training-phase pretrain
    --num-latent-frames $F
    --max-latent-height $H
    --max-latent-width $W
    --max-text-length 512
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-5
    --min-lr 1e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-06
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --save-interval 500000
    --no-load-optim
    --no-load-rng
    --no-strict-fsdp-dtensor-load
    --finetune
    ${RECOMPUTE_ARGS[@]}
)

# FSDP parallel configuration. Pipeline and context parallelism stay disabled.
#
# Required FSDP flags:
#   --use-megatron-fsdp                              enable Megatron FSDP
#   --data-parallel-sharding-strategy optim_grads_params  ZeRO-3: shard params+grads+optimizer
#   --no-gradient-accumulation-fusion                required with FSDP (from doc)
#   --use-distributed-optimizer                      required by --use-precision-aware-optimizer
#   --ckpt-format fsdp_dtensor                       mandatory FSDP checkpoint format
MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CP_SIZE:-1}
    --context-parallel-ulysses-degree ${CP_ULYSSES_DEGREE:-1}
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --ckpt-format fsdp_dtensor
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --distributed-backend nccl
    --attention-backend fused
    --suggested-communication-unit-size 200000000
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

HIGH_NOISE_LOAD_SAVE_ARGS=(
    --load $HIGH_NOISE_CHECKPOINT_PATH
    --save $HIGH_NOISE_CHECKPOINT_PATH
)

HIGH_NOISE_TIMESTEP_BOUNDARY=(
    --max-timestep-boundary 0.358
    --min-timestep-boundary 0.0
)

# Train the high noise model of wan2.2 I2V
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${HIGH_NOISE_LOAD_SAVE_ARGS[@]} \
    ${HIGH_NOISE_TIMESTEP_BOUNDARY[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}

LOW_NOISE_LOAD_SAVE_ARGS=(
    --load $LOW_NOISE_CHECKPOINT_PATH
    --save $LOW_NOISE_CHECKPOINT_PATH
)

LOW_NOISE_TIMESTEP_BOUNDARY=(
    --max-timestep-boundary 1
    --min-timestep-boundary 0.358
)

# Train the low noise model of wan2.2 I2V
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOW_NOISE_LOAD_SAVE_ARGS[@]} \
    ${LOW_NOISE_TIMESTEP_BOUNDARY[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
