#! /bin/bash
# HF Checkpoint Roundtrip Test — Qwen3-30B-A3B

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARNING
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/loongforge-ckpt/Qwen3-30B-A3B"}
SAVE_HF_PATH=${SAVE_HF_PATH:-"/workspace/loongforge-ckpt/qwen3-30b-a3b-roundtrip-output"}

GPUS_PER_NODE=8

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
    --model-name qwen3-30b-a3b
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
    --max-position-embeddings 40960
    --micro-batch-size 1
    --global-batch-size 4
    --bf16
    --norm-epsilon 1e-6
    # --- roundtrip-specific ---
    --train-iters 0
    --no-load-optim
    --no-load-rng
    --load $TOKENIZER_PATH
    --save-hf-path $SAVE_HF_PATH
    --save-hf=true
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
    --attention-backend fused
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --distributed-backend nccl
    --sequence-parallel
)

echo "========================================"
echo "HF Roundtrip Test — Qwen3-30B-A3B"
echo "  Source : $TOKENIZER_PATH"
echo "  Output : $SAVE_HF_PATH"
echo "========================================"

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
    ${MODEL_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]}