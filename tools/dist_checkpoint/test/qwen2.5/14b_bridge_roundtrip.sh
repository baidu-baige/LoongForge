#! /bin/bash
# HF Checkpoint Roundtrip Test for Qwen2.5-14B

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARNING

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/loongforge-ckpt/Qwen2.5-14B-Instruct/"}
SAVE_HF_PATH=${SAVE_HF_PATH:-"/workspace/loongforge-ckpt/qwen2.5-14b-roundtrip-output"}

GPUS_PER_NODE=4

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12345"}
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
    --model-name qwen2.5-14b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)

TOKENIZER_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 4096
    --max-position-embeddings 32768
    --micro-batch-size 1
    --global-batch-size 2
    --bf16
    --norm-epsilon 1e-6
    --train-iters 0
    --no-load-optim
    --no-load-rng
    --load $TOKENIZER_PATH
    --save-hf-path $SAVE_HF_PATH
    --save-hf=true
)

# For 14B model: tp_size=2, pp_size=1
MODEL_PARALLEL_ARGS=(
    --attention-backend fused
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --distributed-backend nccl
)

echo "========================================"
echo "HF Roundtrip Test for Qwen2.5-14B"
echo "  Source : $TOKENIZER_PATH"
echo "  Output : $SAVE_HF_PATH"
echo "========================================"

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
    ${MODEL_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]}