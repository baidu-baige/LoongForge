#! /bin/bash
# HF Checkpoint Roundtrip Test
# Based on bridge_debug.sh — removes training loop, adds roundtrip comparison.
#
# Usage:
#   bash bridge_roundtrip.sh
#
# What it does:
#   1. Builds the Megatron model (same as training)
#   2. Loads the HF checkpoint into the model (load_hf_checkpoint_online)
#   3. Saves model weights back to HF format  (save_hf_checkpoint_online)
#   4. Compares original vs roundtripped weights tensor-by-tensor
#   Report is written to $SAVE_HF_PATH/roundtrip_comparison.json

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARNING

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/loongforge-ckpt/InternVL2_5-38B/"}
SAVE_HF_PATH=${SAVE_HF_PATH:-"/workspace/loongforge-ckpt/internvl2.5-38b-roundtrip-output"}

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
    --model-name internvl2.5-38b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)

# Tokenizer is needed by initialize_loongforge_megatron → set_loongforge_extra_global_vars
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
    # --- roundtrip-specific ---
    --train-iters 0          # no training, only load + save
    --no-load-optim          # skip optimizer state
    --no-load-rng            # skip RNG state
    --load $TOKENIZER_PATH   # original HF checkpoint
    --save-hf-path $SAVE_HF_PATH
    --save-hf=true
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 4
    # --encoder-tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --distributed-backend nccl
)

echo "========================================"
echo "HF Roundtrip Test"
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
