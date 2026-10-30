#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run_giga_brain07_torchrun_8gpu.sh — 8×A800 FSDP2 variant for perf/bs sweep
# Derived from run_giga_brain07_torchrun.sh (kept intact). Adds:
#   - nproc_per_node=8, CUDA_VISIBLE_DEVICES=0..7
#   - --distributed-strategy fsdp  (FSDP2 fully_shard)
#   - --fsdp-wrap-modules Gemma2DecoderLayerWithExpert (one unit per decoder layer)
#   - --activation-checkpoint-module-patterns -> first 15 decoder layers
#     (matches giga single-GPU 'Gemma2DecoderLayerWithExpert__##__15' AC)
#   - env knobs: GB0_DTYPE, GB0_BATCH_SIZE, GB0_NUM_WORKERS, GB0_MAX_STEPS,
#     GB0_EXTRA_ARGS (extra CLI, e.g. bf16 mixed-precision flags), OUTPUT_DIR
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

GB_MODEL_DIR="$LOONGFORGE_PATH/loongforge/embodied/model/giga_brain07"
GB_DATA_DIR="$LOONGFORGE_PATH/loongforge/embodied/data/datasets/giga_brain07"
export PYTHONPATH="$GB_MODEL_DIR:$GB_DATA_DIR:$LOONGFORGE_PATH:${PYTHONPATH:-}"

export GB0_PRETRAINED="${GB0_PRETRAINED:-/workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base}"
export GB0_TOKENIZER="${GB0_TOKENIZER:-/workspace/ckpt/paligemma-3b-pt-224}"
export GB0_FAST_TOKENIZER="${GB0_FAST_TOKENIZER:-/workspace/ckpt/physical-intelligence/fast}"
export GB0_DATA_PATH_V30="${GB0_DATA_PATH_V30:-/workspace/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327}"
export GB0_NORM_STATS="${GB0_NORM_STATS:-/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/LoongForge/experiments/giga_brain07_planB/sweep8gpu}"
GB0_COMPILE=decoder,vision GB0_KV_SLICE=1 GB0_BATCH_SIZE=96 \
GB0_MAX_STEPS="${GB0_MAX_STEPS:-12}"
GB0_NUM_WORKERS="${GB0_NUM_WORKERS:-8}"
GB0_ENABLE_IMAGE_AUG="${GB0_ENABLE_IMAGE_AUG:-false}"
GB0_BATCH_SIZE="${GB0_BATCH_SIZE:-10}"
GB0_DTYPE="${GB0_DTYPE:-float32}"
GB0_AC_PATTERNS="${GB0_AC_PATTERNS:-paligemma_with_expert.layers.0,paligemma_with_expert.layers.1,paligemma_with_expert.layers.2,paligemma_with_expert.layers.3,paligemma_with_expert.layers.4,paligemma_with_expert.layers.5,paligemma_with_expert.layers.6,paligemma_with_expert.layers.7,paligemma_with_expert.layers.8,paligemma_with_expert.layers.9,paligemma_with_expert.layers.10,paligemma_with_expert.layers.11,paligemma_with_expert.layers.12,paligemma_with_expert.layers.13,paligemma_with_expert.layers.14,paligemma_with_expert.layers.15,paligemma_with_expert.layers.16,paligemma_with_expert.layers.17,paligemma_with_expert.layers.18,paligemma_with_expert.layers.19,paligemma_with_expert.layers.20,paligemma_with_expert.layers.21,paligemma_with_expert.layers.22,paligemma_with_expert.layers.23,paligemma_with_expert.layers.24,paligemma_with_expert.layers.25,paligemma_with_expert.vision_tower.encoder.layers.*}"
GB0_FSDP_WRAP="${GB0_FSDP_WRAP:-}"
GB0_AC_SKIP="${GB0_AC_SKIP:-}"
GB0_COMPILE="${GB0_COMPILE:-}"
GB0_COMPILE_MODE="${GB0_COMPILE_MODE:-default}"
GB0_COMPILE_DYNAMIC="${GB0_COMPILE_DYNAMIC:-false}"
GB0_COMPILE_FULLGRAPH="${GB0_COMPILE_FULLGRAPH:-false}"
# Keep Inductor's fused kernels rounding like eager at bf16 down/up-casts (the
# decoder runs under autocast(bf16) in the vendored model); measured to pull
# the compiled step-1..6 loss back inside the eager run-to-run band at ~0 cost.
GB0_COMPILE_EMULATE_CASTS="${GB0_COMPILE_EMULATE_CASTS:-1}"
GB0_COMPILE_ARGS=()
# Measured best fp32 throughput config (8xA800, 2026-10-27, ~73 samples/s vs 31.7 default):
GB0_KV_SLICE="${GB0_KV_SLICE:-}"
if [ -n "$GB0_KV_SLICE" ] && [ "$GB0_KV_SLICE" != 0 ]; then
    GB0_COMPILE_ARGS+=(model.decoder_kv_slice=true)
fi
if [ -n "$GB0_COMPILE" ]; then
    export TORCHINDUCTOR_EMULATE_PRECISION_CASTS="$GB0_COMPILE_EMULATE_CASTS"
    GB0_COMPILE_ARGS+=(
        model.compile_model=true
        "model.compile_scope=$GB0_COMPILE"
        "model.compile_mode=$GB0_COMPILE_MODE"
        "model.compile_dynamic=$GB0_COMPILE_DYNAMIC"
        "model.compile_fullgraph=$GB0_COMPILE_FULLGRAPH"
    )
fi
GB0_EXTRA_ARGS="${GB0_EXTRA_ARGS:-}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZER_PATH="$GB0_TOKENIZER"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29533}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

mkdir -p "$OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"
echo "  GigaBrain-0.7 (方案B) — 8×A800 FSDP2 sweep entry"
echo "  gpus=$GPUS_PER_NODE dtype=$GB0_DTYPE bs/gpu=$GB0_BATCH_SIZE steps=$GB0_MAX_STEPS workers=$GB0_NUM_WORKERS"
echo "  ac_skip='$GB0_AC_SKIP'"
echo "  compile='$GB0_COMPILE' kv_slice='$GB0_KV_SLICE' mode=$GB0_COMPILE_MODE dynamic=$GB0_COMPILE_DYNAMIC fullgraph=$GB0_COMPILE_FULLGRAPH emulate_casts=${TORCHINDUCTOR_EMULATE_PRECISION_CASTS:-unset}"
echo "  extra='$GB0_EXTRA_ARGS'"
echo "  output=$OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$PYTHONPATH" torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    --model-name giga_brain07 \
    --trainer-type GigaBrain07Trainer \
    --train-iters "$GB0_MAX_STEPS" \
    --per-device-batch-size "$GB0_BATCH_SIZE" \
    --gradient-accumulation-steps 1 \
    --seed 6666 \
    --output-dir "$OUTPUT_DIR" \
    --distributed-strategy fsdp \
    ${GB0_FSDP_WRAP:+--fsdp-wrap-modules $GB0_FSDP_WRAP} \
    --activation-checkpoint-module-patterns "$GB0_AC_PATTERNS" \
    ${GB0_AC_SKIP:+--activation-checkpoint-skip-modules "$GB0_AC_SKIP"} \
    --dtype "$GB0_DTYPE" \
    $GB0_EXTRA_ARGS \
    --optimizer AdamW \
    --clip-grad 0 \
    --loss-spike-threshold 1000 \
    --save-interval 0 \
    --log-interval 1 \
    --num-workers "$GB0_NUM_WORKERS" \
    --wandb-mode disabled \
    --dataset-path "$GB0_DATA_PATH_V30" \
    --tokenizer-path "$GB0_TOKENIZER" \
    model.pretrained="$GB0_PRETRAINED" \
    data.data_path="$GB0_DATA_PATH_V30" \
    data.norm_stats_path="$GB0_NORM_STATS" \
    data.tokenizer_model_path="$GB0_TOKENIZER" \
    data.fast_tokenizer_path="$GB0_FAST_TOKENIZER" \
    data.num_workers="$GB0_NUM_WORKERS" \
    data.enable_image_aug="$GB0_ENABLE_IMAGE_AUG" \
    data.batch_size_per_gpu="$GB0_BATCH_SIZE" \
    ${GB0_COMPILE_ARGS[@]+"${GB0_COMPILE_ARGS[@]}"}
