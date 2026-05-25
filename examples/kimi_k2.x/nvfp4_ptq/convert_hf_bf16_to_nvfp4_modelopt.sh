#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checkpoint and tool paths.
BF16_HF=${BF16_HF:?Set BF16_HF to the Kimi K2.x BF16 HuggingFace checkpoint directory.}
SAVE=${SAVE:?Set SAVE to the Kimi K2.x ModelOpt NVFP4 HuggingFace output directory.}
MODELOPT_REPO=${MODELOPT_REPO:?Set MODELOPT_REPO to a Model-Optimizer checkout. Run the matching Kimi version install_nvfp4_modelopt_deps.sh first.}
PTQ_DRIVER=${PTQ_DRIVER:-"${SCRIPT_DIR}/kimi_k2x_nvfp4_modelopt_driver.py"}
OFFICIAL_QUANT_CONFIG=${OFFICIAL_QUANT_CONFIG:?Set OFFICIAL_QUANT_CONFIG to the Kimi K2.x ModelOpt quant config JSON.}

# Quantization recipe selection.
QFORMAT=${QFORMAT:-"nvfp4_mlp_only"}
KV_CACHE_QFORMAT=${KV_CACHE_QFORMAT:-"fp8"}

# Calibration data and batch sizing.
CALIB_DATASET=${CALIB_DATASET:-"cnn_dailymail"}
CALIB_SIZE=${CALIB_SIZE:-1}
CALIB_SEQ=${CALIB_SEQ:-512}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-0}
CALIB_DATASET_JSONL=${CALIB_DATASET_JSONL:-}

# Model placement and memory behavior.
GPU_MAX_MEM_PERCENTAGE=${GPU_MAX_MEM_PERCENTAGE:-0.8}
INFERENCE_TENSOR_PARALLEL=${INFERENCE_TENSOR_PARALLEL:-1}
INFERENCE_PIPELINE_PARALLEL=${INFERENCE_PIPELINE_PARALLEL:-1}
LOW_MEMORY_MODE=${LOW_MEMORY_MODE:-0}
USE_SEQ_DEVICE_MAP=${USE_SEQ_DEVICE_MAP:-1}
MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT=${MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT:-0}

# Runtime compatibility switches.
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-"eager"}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
SKIP_GENERATE=${SKIP_GENERATE:-1}
ENABLE_HF_PARALLEL_LOADING=${ENABLE_HF_PARALLEL_LOADING:-1}
HF_PARALLEL_LOADING_WORKERS=${HF_PARALLEL_LOADING_WORKERS:-8}

# Optional MoE calibration diagnostics.
CALIBRATE_ALL_MOE_EXPERTS=${CALIBRATE_ALL_MOE_EXPERTS:-0}
MOE_ALL_EXPERTS_MAX_TOKENS=${MOE_ALL_EXPERTS_MAX_TOKENS:-1}
MOE_ALL_EXPERTS_EVERY_FORWARD=${MOE_ALL_EXPERTS_EVERY_FORWARD:-0}

# Logging and output safety.
VERBOSE=${VERBOSE:-1}
FORCE_EXISTING_OUTPUT=${FORCE_EXISTING_OUTPUT:-0}
SCRIPT_START_EPOCH=$(date +%s)
SCRIPT_START_TIME=$(date '+%F %T')

mkdir -p "$SAVE/logs"
LOG=${LOG:-"$SAVE/logs/convert_nvfp4_$(date +%Y%m%d-%H%M%S).log"}
exec &> >(tee "$LOG")

finish_timer() {
    local status=$?
    local end_epoch end_time elapsed h m s
    end_epoch=$(date +%s)
    end_time=$(date '+%F %T')
    elapsed=$((end_epoch - SCRIPT_START_EPOCH))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))

    echo ""
    echo "===== BF16 HuggingFace -> ModelOpt NVFP4 conversion finished ====="
    echo ">> START_TIME:             $SCRIPT_START_TIME"
    echo ">> END_TIME:               $end_time"
    echo ">> EXIT_STATUS:            $status"
    echo ">> ELAPSED_SECONDS:        $elapsed"
    printf ">> ELAPSED_HHMMSS:         %02d:%02d:%02d\n" "$h" "$m" "$s"

    exit "$status"
}
trap finish_timer EXIT

echo "===== BF16 HuggingFace -> ModelOpt NVFP4 conversion ($(date '+%F %T')) ====="
echo ">> LOG:                    $LOG"
echo ">> START_TIME:             $SCRIPT_START_TIME"
echo ">> BF16_HF:                $BF16_HF"
echo ">> SAVE:                   $SAVE"
echo ">> MODELOPT_REPO:          $MODELOPT_REPO"
if [ -d "$MODELOPT_REPO/.git" ]; then
    echo ">> MODELOPT_REPO_REF:      $(git -C "$MODELOPT_REPO" branch --show-current 2>/dev/null || true)"
    echo ">> MODELOPT_REPO_COMMIT:   $(git -C "$MODELOPT_REPO" rev-parse HEAD)"
fi
echo ">> PTQ_DRIVER:             $PTQ_DRIVER"
echo ">> OFFICIAL_QUANT_CONFIG:  $OFFICIAL_QUANT_CONFIG"
echo ">> QFORMAT:                $QFORMAT"
echo ">> KV_CACHE_QFORMAT:       $KV_CACHE_QFORMAT"
echo ">> CALIB_DATASET:          $CALIB_DATASET"
echo ">> CALIB_SIZE:             $CALIB_SIZE"
echo ">> CALIB_SEQ:              $CALIB_SEQ"
echo ">> CALIB_BATCH_SIZE:       $CALIB_BATCH_SIZE"
echo ">> CALIB_DATASET_JSONL:    ${CALIB_DATASET_JSONL:-<required>}"
echo ">> INFERENCE_TENSOR_PARALLEL: $INFERENCE_TENSOR_PARALLEL"
echo ">> INFERENCE_PIPELINE_PARALLEL: $INFERENCE_PIPELINE_PARALLEL"
echo ">> ATTN_IMPLEMENTATION:    ${ATTN_IMPLEMENTATION:-<config default>}"
echo ">> LOW_MEMORY_MODE:        $LOW_MEMORY_MODE"
echo ">> USE_SEQ_DEVICE_MAP:     $USE_SEQ_DEVICE_MAP"
echo ">> MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT: $MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT"
echo ">> TRUST_REMOTE_CODE:      $TRUST_REMOTE_CODE"
echo ">> SKIP_GENERATE:          $SKIP_GENERATE"
echo ">> ENABLE_HF_PARALLEL_LOADING: $ENABLE_HF_PARALLEL_LOADING"
echo ">> HF_PARALLEL_LOADING_WORKERS: $HF_PARALLEL_LOADING_WORKERS"
echo ">> CALIBRATE_ALL_MOE_EXPERTS: $CALIBRATE_ALL_MOE_EXPERTS"
echo ">> MOE_ALL_EXPERTS_MAX_TOKENS: $MOE_ALL_EXPERTS_MAX_TOKENS"
echo ">> MOE_ALL_EXPERTS_EVERY_FORWARD: $MOE_ALL_EXPERTS_EVERY_FORWARD"
echo ""

case "$QFORMAT" in
    nvfp4 | nvfp4_mlp_only | nvfp4_awq) ;;
    *)
        echo "Unsupported QFORMAT=$QFORMAT. Expected one of: nvfp4, nvfp4_mlp_only, nvfp4_awq" >&2
        exit 1
        ;;
esac

if [ "$QFORMAT" != "nvfp4_mlp_only" ]; then
    echo "WARNING: NVIDIA Kimi NVFP4 keeps self-attention and lm_head out of FP4."
    echo "         The default QFORMAT=nvfp4_mlp_only is the closest ModelOpt preset."
fi
if [ "$CALIBRATE_ALL_MOE_EXPERTS" = "1" ]; then
    echo "WARNING: CALIBRATE_ALL_MOE_EXPERTS=1 forces extra all-expert MoE forwards during calibration."
    echo "         This is extremely slow for full-size Kimi and should only be used for diagnosis."
    if [ "$MOE_ALL_EXPERTS_EVERY_FORWARD" = "1" ]; then
        echo "WARNING: MOE_ALL_EXPERTS_EVERY_FORWARD=1 repeats that all-expert path for every sample."
    fi
fi

if [ ! -d "$BF16_HF" ]; then
    echo "BF16_HF does not exist or is not a directory: $BF16_HF" >&2
    exit 1
fi
if [ ! -f "$BF16_HF/config.json" ]; then
    echo "Missing config.json under BF16_HF: $BF16_HF" >&2
    exit 1
fi
if [ ! -f "$BF16_HF/model.safetensors.index.json" ]; then
    echo "Missing model.safetensors.index.json under BF16_HF: $BF16_HF" >&2
    exit 1
fi
if [ ! -f "$PTQ_DRIVER" ]; then
    echo "Missing Kimi ModelOpt PTQ driver: $PTQ_DRIVER" >&2
    exit 1
fi
if [ ! -f "$OFFICIAL_QUANT_CONFIG" ]; then
    echo "Missing NVIDIA Kimi NVFP4 recipe config: $OFFICIAL_QUANT_CONFIG" >&2
    exit 1
fi
if [ -z "$CALIB_DATASET_JSONL" ]; then
    echo "Set CALIB_DATASET_JSONL to an existing prefetched calibration dataset JSONL." >&2
    echo "This conversion script does not download datasets; run the matching Kimi version's nvfp4_ptq/download_calib_dataset.sh separately if needed." >&2
    exit 1
fi
if [ ! -f "$CALIB_DATASET_JSONL" ]; then
    echo "Calibration dataset JSONL does not exist: $CALIB_DATASET_JSONL" >&2
    echo "This conversion script does not download datasets; run the matching Kimi version's nvfp4_ptq/download_calib_dataset.sh separately if needed." >&2
    exit 1
fi

PTQ_INPUT=${PTQ_INPUT:-"$BF16_HF"}
echo ">> PTQ_INPUT:              $PTQ_INPUT"

export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"$SAVE/.hf_modules_cache"}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TRANSFORMERS_NO_TORCHVISION=${TRANSFORMERS_NO_TORCHVISION:-1}
export MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT
if [ "$ENABLE_HF_PARALLEL_LOADING" = "1" ]; then
    export HF_ENABLE_PARALLEL_LOADING=true
    export HF_PARALLEL_LOADING_WORKERS
else
    unset HF_ENABLE_PARALLEL_LOADING
fi
mkdir -p "$HF_MODULES_CACHE"
echo ">> HF_MODULES_CACHE:       $HF_MODULES_CACHE"
echo ">> PYTHONUNBUFFERED:       $PYTHONUNBUFFERED"
echo ">> TRANSFORMERS_NO_TORCHVISION: $TRANSFORMERS_NO_TORCHVISION"
echo ">> HF_ENABLE_PARALLEL_LOADING: ${HF_ENABLE_PARALLEL_LOADING:-<unset>}"
echo ">> HF_PARALLEL_LOADING_WORKERS(env): ${HF_PARALLEL_LOADING_WORKERS:-<unset>}"

existing_artifacts=$(find "$SAVE" -maxdepth 1 \( \
    -name 'model-*.safetensors' -o \
    -name 'model.safetensors' -o \
    -name 'model.safetensors.index.json' -o \
    -name 'config.json' -o \
    -name 'hf_quant_config.json' \
    \) -print -quit)

if [ -n "$existing_artifacts" ]; then
    if [ "$FORCE_EXISTING_OUTPUT" != "1" ]; then
        echo "Output already contains ModelOpt artifacts: $existing_artifacts" >&2
        echo "Use FORCE_EXISTING_OUTPUT=1 to delete known output artifacts and rerun." >&2
        exit 1
    fi

    echo "FORCE_EXISTING_OUTPUT=1: deleting known output artifacts under $SAVE"
    find "$SAVE" -maxdepth 1 \( \
        -name 'model-*.safetensors' -o \
        -name 'model.safetensors' -o \
        -name 'model.safetensors.index.json' -o \
        -name 'config.json' -o \
        -name 'hf_quant_config.json' -o \
        -name 'generation_config.json' -o \
        -name 'chat_template.jinja' -o \
        -name 'preprocessor_config.json' -o \
        -name 'processor_config.json' -o \
        -name 'special_tokens_map.json' -o \
        -name 'tiktoken.model' -o \
        -name 'tokenization_*.py' -o \
        -name 'tokenizer*' -o \
        -name '.quant_summary.txt' -o \
        -name '*.py' \
        \) -type f -delete
fi

PTQ_ARGS=(
    --modelopt_repo "$MODELOPT_REPO"
    --official_quant_config "$OFFICIAL_QUANT_CONFIG"
    --pyt_ckpt_path "$PTQ_INPUT"
    --export_path "$SAVE"
    --qformat "$QFORMAT"
    --kv_cache_qformat "$KV_CACHE_QFORMAT"
    --dataset "$CALIB_DATASET"
    --calib_size "$CALIB_SIZE"
    --calib_dataset_jsonl "$CALIB_DATASET_JSONL"
    --calib_seq "$CALIB_SEQ"
    --batch_size "$CALIB_BATCH_SIZE"
    --gpu_max_mem_percentage "$GPU_MAX_MEM_PERCENTAGE"
    --inference_tensor_parallel "$INFERENCE_TENSOR_PARALLEL"
    --inference_pipeline_parallel "$INFERENCE_PIPELINE_PARALLEL"
)

if [ "$LOW_MEMORY_MODE" = "1" ]; then
    PTQ_ARGS+=(--low_memory_mode)
fi
if [ "$USE_SEQ_DEVICE_MAP" = "1" ]; then
    PTQ_ARGS+=(--use_seq_device_map)
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    PTQ_ARGS+=(--trust_remote_code)
fi
if [ "$SKIP_GENERATE" = "1" ]; then
    PTQ_ARGS+=(--skip_generate)
fi
if [ "$CALIBRATE_ALL_MOE_EXPERTS" = "1" ]; then
    PTQ_ARGS+=(--calibrate_all_moe_experts)
    PTQ_ARGS+=(--moe_all_experts_max_tokens "$MOE_ALL_EXPERTS_MAX_TOKENS")
    if [ "$MOE_ALL_EXPERTS_EVERY_FORWARD" = "1" ]; then
        PTQ_ARGS+=(--moe_all_experts_every_forward)
    fi
fi
if [ -n "$ATTN_IMPLEMENTATION" ]; then
    PTQ_ARGS+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi
if [ "$VERBOSE" != "1" ]; then
    PTQ_ARGS+=(--no-verbose)
fi

echo ""
echo "===== Running local Kimi ModelOpt PTQ driver ====="
printf ' %q' python "$PTQ_DRIVER" "${PTQ_ARGS[@]}"
echo ""

python "$PTQ_DRIVER" "${PTQ_ARGS[@]}"

missing_artifacts=()
for artifact in config.json model.safetensors.index.json hf_quant_config.json; do
    if [ ! -f "$SAVE/$artifact" ]; then
        missing_artifacts+=("$artifact")
    fi
done

if [ "${#missing_artifacts[@]}" -ne 0 ]; then
    echo "ModelOpt driver exited without expected NVFP4 checkpoint artifacts under $SAVE." >&2
    echo "Missing: ${missing_artifacts[*]}" >&2
    exit 1
fi

if ! find "$SAVE" -maxdepth 1 -name 'model-*.safetensors' -print -quit | grep -q .; then
    echo "ModelOpt driver exited without model-*.safetensors shards under $SAVE." >&2
    exit 1
fi

cat <<EOF
Saved ModelOpt NVFP4 HuggingFace checkpoint to: $SAVE

The checkpoint is intended for ModelOpt-aware runtimes such as vLLM/TensorRT-LLM
on Blackwell-class GPUs. Conversion log: $LOG
EOF
