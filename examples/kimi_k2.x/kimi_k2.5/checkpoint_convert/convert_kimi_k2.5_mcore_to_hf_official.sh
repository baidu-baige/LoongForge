#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"
TORCHRUN=${TORCHRUN:-torchrun}

OFFICIAL_HF_PATH=${OFFICIAL_HF_PATH:-"/mnt/cluster/huggingface.co/moonshotai/Kimi-K2.5"}
OFFICIAL_CONFIG_FILE=${OFFICIAL_CONFIG_FILE:-"${OFFICIAL_HF_PATH}/config.json"}

LOAD=${LOAD:-"/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-entp1dtp8pp8ep16etp1"}
SAVE=${SAVE:-"/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-hf-official"}

MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-"${LOONGFORGE_PATH}/configs/models/kimi_k2.5/kimi_k2_5.yaml"}

FOUNDATION_CONVERT_FILE=${FOUNDATION_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/kimi_k2/ckpt_convert/kimi_k2_convert.yaml"}
IMAGE_ENCODER_CONVERT_FILE=${IMAGE_ENCODER_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/moon_vit_3d_convert.yaml"}
IMAGE_PROJECTOR_CONVERT_FILE=${IMAGE_PROJECTOR_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/patch_merger_adapter_convert.yaml"}

ENCODER_TP=${ENCODER_TP:-1}
DTP=${DTP:-8}
PP=${PP:-8}
EP=${EP:-16}
EXPERT_TP=${EXPERT_TP:-1}
VPP=${VPP:-2}
CUSTOM_PIPELINE_LAYERS=${CUSTOM_PIPELINE_LAYERS:-"4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,2"}

# This script always uses torchrun with --distributed_convert. Defaults are for
# one node; set NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT in each pod for multinode.
MAX_WORKERS=${MAX_WORKERS:-4}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

KIMI_EXPERT_TARGET_REGEX='^language_model\.model\.layers\.[0-9]+\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$'

mkdir -p "$SAVE/logs"
LOG=${LOG:-"$SAVE/logs/convert_node${NODE_RANK}_$(date +%Y%m%d-%H%M%S).log"}
exec &> >(tee "$LOG")

echo "===== Kimi K2.5 full M-Core -> official HuggingFace conversion ($(date '+%F %T')) ====="
echo ">> LOG:                $LOG"
echo ">> LOAD:               $LOAD"
echo ">> SAVE:               $SAVE"
echo ">> OFFICIAL_HF_PATH:   $OFFICIAL_HF_PATH"
echo ">> OFFICIAL_CONFIG:    $OFFICIAL_CONFIG_FILE"
echo ">> MODEL_CONFIG:       $MODEL_CONFIG_FILE"
echo ""

if [[ ! -d "$LOAD" ]]; then
    echo "Missing M-Core checkpoint directory: $LOAD" >&2
    exit 1
fi

if [[ ! -f "$OFFICIAL_CONFIG_FILE" ]]; then
    echo "Missing official Kimi K2.5 config file: $OFFICIAL_CONFIG_FILE" >&2
    echo "Set OFFICIAL_HF_PATH or OFFICIAL_CONFIG_FILE to a local Kimi K2.5 HuggingFace metadata directory." >&2
    exit 1
fi

echo ">> TORCHRUN:           $TORCHRUN"
echo ">> NNODES:             $NNODES"
echo ">> NPROC_PER_NODE:     $NPROC_PER_NODE"
echo ">> NODE_RANK:          $NODE_RANK"
echo ">> MASTER_ADDR:        $MASTER_ADDR"
echo ">> MASTER_PORT:        $MASTER_PORT"
echo ">> MAX_WORKERS/rank:   $MAX_WORKERS"
echo ""

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    "$TORCHRUN" \
    --nnodes "$NNODES" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$CONVERT_CHECKPOINT_PATH/module_convertor/model.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file "$MODEL_CONFIG_FILE" \
    --convert_file "$FOUNDATION_CONVERT_FILE" \
    --adapter_convert_file "$IMAGE_PROJECTOR_CONVERT_FILE" \
    --vision_patch_convert_file "$IMAGE_ENCODER_CONVERT_FILE" \
    --encoder_tensor_model_parallel_size="$ENCODER_TP" \
    --tensor_model_parallel_size="$DTP" \
    --pipeline_model_parallel_size="$PP" \
    --expert_parallel_size="$EP" \
    --expert_tensor_parallel_size="$EXPERT_TP" \
    --num-virtual-stages-per-pipeline-rank="$VPP" \
    --custom_pipeline_layers "$CUSTOM_PIPELINE_LAYERS" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE" \
    --enable-full-hetero-dp \
    --safetensors \
    --fp8_force_no_requant \
    --moe-grouped-gemm \
    --no_save_optim \
    --no_load_optim \
    --hf-official-config-file "$OFFICIAL_CONFIG_FILE" \
    --hf-pack-quantized-from-config \
    --hf-pack-quantized-target-regex "$KIMI_EXPERT_TARGET_REGEX" \
    --distributed_convert \
    --max_workers "$MAX_WORKERS"

# Complete the output as a usable HuggingFace repo by copying the official
# metadata, tokenizer, processor, and remote-code files. Weight shards and
# weight index files are skipped so the converted checkpoint is not overwritten.
if [[ "$NODE_RANK" == "0" ]]; then
python - "$OFFICIAL_HF_PATH" "$SAVE" <<'PY'
import shutil
import sys
from pathlib import Path

official_hf_path = Path(sys.argv[1])
save_path = Path(sys.argv[2])

save_path.mkdir(parents=True, exist_ok=True)

skip_prefixes = ("model-", "pytorch_model")
skip_names = {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin.index.json"}
for src in official_hf_path.iterdir():
    if not src.is_file():
        continue
    if src.name in skip_names or src.name.startswith(skip_prefixes):
        continue
    shutil.copy2(src, save_path / src.name)
PY
else
    echo "Skip HuggingFace metadata copy on NODE_RANK=$NODE_RANK."
fi

echo "Saved official-format HuggingFace checkpoint to: $SAVE"
