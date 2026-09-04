#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
MEGATRON_PATH=${MEGATRON_PATH:-/workspace/Loong-Megatron}
HF_MODEL_PATH=${HF_MODEL_PATH:?Set HF_MODEL_PATH to the Kimi K3 Hugging Face checkpoint}
MCORE_SAVE_PATH=${MCORE_SAVE_PATH:?Set MCORE_SAVE_PATH for the converted checkpoint}

FOUNDATION_CONVERT_FILE=${FOUNDATION_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/kimi_k3/ckpt_convert/kimi_k3_convert.yaml"}
IMAGE_ENCODER_CONVERT_FILE=${IMAGE_ENCODER_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/image_encoder/ckpt_convert/kimi_k3_vit_convert.yaml"}
IMAGE_PROJECTOR_CONVERT_FILE=${IMAGE_PROJECTOR_CONVERT_FILE:-"$LOONGFORGE_PATH/configs/models/image_projector/ckpt_convert/kimi_k3_patch_merger_convert.yaml"}

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}

PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/tools/convert_checkpoint/module_convertor/model.py" \
  --load_platform=huggingface \
  --save_platform=mcore \
  --config_file "$LOONGFORGE_PATH/configs/models/kimi_k3/kimi_k3.yaml" \
  --convert_file "$FOUNDATION_CONVERT_FILE" \
  --adapter_convert_file "$IMAGE_PROJECTOR_CONVERT_FILE" \
  --vision_patch_convert_file "$IMAGE_ENCODER_CONVERT_FILE" \
  --tensor_model_parallel_size="$TP" \
  --pipeline_model_parallel_size="$PP" \
  --expert_parallel_size="$EP" \
  --expert_tensor_parallel_size="$ETP" \
  --megatron_path="$MEGATRON_PATH" \
  --load_ckpt_path="$HF_MODEL_PATH" \
  --save_ckpt_path="$MCORE_SAVE_PATH" \
  --safetensors \
  --max_workers="${MAX_WORKERS:-32}" \
  --moe-grouped-gemm
