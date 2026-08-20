#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
MEGATRON_PATH=${MEGATRON_PATH:-"${LOONGFORGE_PATH}/third_party/Loong-Megatron"}
CONVERTER_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=${LOAD:?Set LOAD to the Hugging Face checkpoint directory}
SAVE=${SAVE:?Set SAVE to the destination MCore checkpoint directory}
TMP_DIR=${TMP_DIR:-"${SAVE}/tmp"}

SAVE_LANGUAGE_MODEL="${TMP_DIR}/language-mcore"
SAVE_VISION_MODEL="${TMP_DIR}/vision-mcore"
SAVE_VISION_PATCH="${TMP_DIR}/vision-patch-mcore"
SAVE_ADAPTER="${TMP_DIR}/adapter-mcore"

MODEL_CONFIG_FILE="${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml"
FOUNDATION_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/ckpt_convert/minicpm_v_4_6_llm_convert.yaml"
IMAGE_PROJECTOR_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/minicpm_v_4_6_merger_convert.yaml"
ETP=${ETP:-1}
DTP=${DTP:-1}
PP=${PP:-1}
# The released MiniCPM-V-4.6 checkpoint declares MTP but contains no MTP tensors.
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
MAKE_VOCAB_SIZE_DIVISIBLE_BY=${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-1}

case "${TMP_DIR}" in
    "${SAVE}"/*) ;;
    *) echo "TMP_DIR must be inside SAVE" >&2; exit 2 ;;
esac
rm -rf "${TMP_DIR}"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${FOUNDATION_CONVERT_FILE}" \
    --tensor_model_parallel_size="${DTP}" \
    --pipeline_model_parallel_size="${PP}" \
    --load_ckpt_path="${LOAD}" \
    --save_ckpt_path="${SAVE_LANGUAGE_MODEL}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --make_vocab_size_divisible_by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}" \
    --mtp_num_layers "${MTP_NUM_LAYERS}"

IMAGE_ENCODER_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/minicpm_v_4_6_vit_convert.yaml"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${IMAGE_ENCODER_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --load_ckpt_path="${LOAD}" \
    --save_ckpt_path="${SAVE_VISION_MODEL}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/vision_patch.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${IMAGE_ENCODER_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --load_ckpt_path="${LOAD}" \
    --save_ckpt_path="${SAVE_VISION_PATCH}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/adapter.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file="${MODEL_CONFIG_FILE}" \
    --convert_file="${IMAGE_PROJECTOR_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --load_ckpt_path="${LOAD}" \
    --save_ckpt_path="${SAVE_ADAPTER}"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/mcore/merge_megatron.py" \
    --megatron_path "${MEGATRON_PATH}" \
    --language_model_path "${SAVE_LANGUAGE_MODEL}/release" \
    --vision_model_path "${SAVE_VISION_MODEL}/release" \
    --vision_patch "${SAVE_VISION_PATCH}/release" \
    --adapter_path "${SAVE_ADAPTER}/release" \
    --encoder_tensor_model_parallel_size "${ETP}" \
    --decoder_tensor_model_parallel_size "${DTP}" \
    --pipeline_model_parallel_size "${PP}" \
    --save_ckpt_path "${SAVE}/release" \
    --config_file "${MODEL_CONFIG_FILE}"

echo release > "${SAVE}/latest_checkpointed_iteration.txt"
rm -rf "${TMP_DIR}"
