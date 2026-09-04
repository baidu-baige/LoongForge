#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
MEGATRON_PATH=${MEGATRON_PATH:-/workspace/Loong-Megatron}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"
TORCHRUN=${TORCHRUN:-torchrun}

OFFICIAL_HF_PATH=${OFFICIAL_HF_PATH:?Set OFFICIAL_HF_PATH to Kimi K3 metadata and tokenizer files}
LOAD=${LOAD:?Set LOAD to the MCore checkpoint directory}
SAVE=${SAVE:?Set SAVE to the BF16 HuggingFace output directory}
MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-"$LOONGFORGE_PATH/configs/models/kimi_k3/kimi_k3.yaml"}
FOUNDATION_CONVERT_FILE=${FOUNDATION_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/kimi_k3/ckpt_convert/kimi_k3_convert.yaml"}
IMAGE_ENCODER_CONVERT_FILE=${IMAGE_ENCODER_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/image_encoder/ckpt_convert/kimi_k3_vit_convert.yaml"}
IMAGE_PROJECTOR_CONVERT_FILE=${IMAGE_PROJECTOR_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/image_projector/ckpt_convert/kimi_k3_patch_merger_convert.yaml"}

ENCODER_TP=${ENCODER_TP:-1}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
MAX_WORKERS=${MAX_WORKERS:-4}

if [[ ! -d "$LOAD" || ! -f "$OFFICIAL_HF_PATH/config.json" ]]; then
  echo "LOAD must be an MCore directory and OFFICIAL_HF_PATH must contain config.json" >&2
  exit 1
fi

PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-}" \
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
  --encoder_tensor_model_parallel_size "$ENCODER_TP" \
  --tensor_model_parallel_size "$TP" \
  --pipeline_model_parallel_size "$PP" \
  --expert_parallel_size "$EP" \
  --expert_tensor_parallel_size "$ETP" \
  --load_ckpt_path "$LOAD" \
  --save_ckpt_path "$SAVE" \
  --enable-full-hetero-dp \
  --safetensors \
  --torch_dtype bfloat16 \
  --no_save_optim \
  --no_load_optim \
  --moe-grouped-gemm \
  --distributed_convert \
  --max_workers "$MAX_WORKERS"

if [[ "$NODE_RANK" == "0" ]]; then
  python - "$OFFICIAL_HF_PATH" "$SAVE" <<'PY'
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
skip = {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin.index.json"}
for path in source.iterdir():
    if path.is_file() and path.name not in skip and not path.name.startswith(("model-", "pytorch_model")):
        shutil.copy2(path, target / path.name)
PY
fi

echo "Saved BF16 HuggingFace checkpoint to: $SAVE"
