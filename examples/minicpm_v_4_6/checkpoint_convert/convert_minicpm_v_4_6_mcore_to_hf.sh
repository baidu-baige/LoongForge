#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
MEGATRON_PATH=${MEGATRON_PATH:-"${LOONGFORGE_PATH}/third_party/Loong-Megatron"}
CONVERTER_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=${LOAD:?Set LOAD to the merged MCore release directory}
SAVE=${SAVE:?Set SAVE to the destination Hugging Face directory}
ORIGINAL_HF_PATH=${ORIGINAL_HF_PATH:?Set ORIGINAL_HF_PATH to the source Hugging Face directory}
TMP_DIR=${TMP_DIR:-"${SAVE}/tmp"}

REVERSED_MCORE="${TMP_DIR}/reversed-mcore"
SAVE_LANGUAGE_MODEL="${TMP_DIR}/language-hf"
SAVE_VISION_MODEL="${TMP_DIR}/vision-hf"
SAVE_VISION_PATCH="${TMP_DIR}/vision-patch-hf"
SAVE_ADAPTER="${TMP_DIR}/adapter-hf"

MODEL_CONFIG_FILE="${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml"
FOUNDATION_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/ckpt_convert/minicpm_v_4_6_llm_convert.yaml"
IMAGE_PROJECTOR_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/minicpm_v_4_6_merger_convert.yaml"
ETP=${ETP:-1}
DTP=${DTP:-1}
PP=${PP:-1}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}

case "${TMP_DIR}" in
    "${SAVE}"/*) ;;
    *) echo "TMP_DIR must be inside SAVE" >&2; exit 2 ;;
esac
rm -rf "${TMP_DIR}"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/key_mappings/key_reverser.py" \
    --load_omni_ckpt_path "${LOAD}" \
    --save_original_ckpt_path "${REVERSED_MCORE}" \
    --decoder_tensor_model_parallel_size "${DTP}" \
    --pipeline_model_parallel_size "${PP}" \
    --config_file "${MODEL_CONFIG_FILE}"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/model.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${FOUNDATION_CONVERT_FILE}" \
    --tensor_model_parallel_size="${DTP}" \
    --pipeline_model_parallel_size="${PP}" \
    --load_ckpt_path="${REVERSED_MCORE}" \
    --save_ckpt_path="${SAVE_LANGUAGE_MODEL}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --mtp_num_layers "${MTP_NUM_LAYERS}"

IMAGE_ENCODER_CONVERT_FILE="${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/minicpm_v_4_6_vit_convert.yaml"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/model.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${IMAGE_ENCODER_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path="${REVERSED_MCORE}" \
    --save_ckpt_path="${SAVE_VISION_MODEL}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/vision_patch.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file "${MODEL_CONFIG_FILE}" \
    --convert_file "${IMAGE_ENCODER_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path="${REVERSED_MCORE}" \
    --save_ckpt_path="${SAVE_VISION_PATCH}" \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/module_convertor/adapter.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file="${MODEL_CONFIG_FILE}" \
    --convert_file="${IMAGE_PROJECTOR_CONVERT_FILE}" \
    --tensor_model_parallel_size="${ETP}" \
    --pipeline_model_parallel_size="${PP}" \
    --load_ckpt_path="${REVERSED_MCORE}" \
    --save_ckpt_path="${SAVE_ADAPTER}"

PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    python "${CONVERTER_PATH}/huggingface/merge_huggingface.py" \
    --megatron_path "${MEGATRON_PATH}" \
    --language_model_path "${SAVE_LANGUAGE_MODEL}" \
    --vision_model_path "${SAVE_VISION_MODEL}" \
    --vision_patch "${SAVE_VISION_PATCH}" \
    --adapter_path "${SAVE_ADAPTER}" \
    --save_ckpt_path "${SAVE}"

for path in "${ORIGINAL_HF_PATH}"/*; do
    [ -f "${path}" ] || continue
    case "$(basename "${path}")" in
        model.safetensors|model.safetensors.index.json|model-*.safetensors|pytorch_model*.bin) continue ;;
    esac
    cp -p "${path}" "${SAVE}/"
done

rm -rf "${TMP_DIR}"
