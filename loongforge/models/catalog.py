# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model names, engines, default YAMLs, lazy config type bindings, and Megatron model families."""

from importlib import import_module
from pathlib import Path
from typing import List


MODEL_CONFIG_REGISTRY = {
    # deepseek
    "deepseek-v2": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v2",
        "config_name": "deepseek_v2",
    },
    "deepseek-v2-lite": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v2",
        "config_name": "deepseek_v2_lite",
    },
    "deepseek-v3": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v3",
        "config_name": "deepseek_v3",
    },
    "deepseek-v3.2-sparse": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v3",
        "config_name": "deepseek_v3_2_sparse",
    },
    "deepseek-v4-flash": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_flash_base",
    },
    "deepseek-v4-flash-lite": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_flash_lite",
    },
    "deepseek-v4-flash-lite-2l": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_flash_lite_2l",
    },
    "deepseek-v4-flash-lite-4l": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_flash_lite_4l",
    },
    "deepseek-v4-flash-lite-6l": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_flash_lite_6l",
    },
    "deepseek-v4-pro": {
        "engine": "megatron",
        "config_path": "configs/models/deepseek_v4",
        "config_name": "deepseek_v4_pro_base",
    },
    # internlm2.5
    "internlm2.5-8b": {
        "engine": "megatron",
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_8b",
    },
    "internlm2.5-20b": {
        "engine": "megatron",
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_20b",
    },
    # llama
    "llama2-7b": {
        "engine": "megatron",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_7b",
    },
    "llama2-13b": {
        "engine": "megatron",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_13b",
    },
    "llama2-70b": {
        "engine": "megatron",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_70b",
    },
    "llama3-8b": {
        "engine": "megatron",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_8b",
    },
    "llama3-70b": {
        "engine": "megatron",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_70b",
    },
    "llama3.1-8b": {
        "engine": "megatron",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_8b",
    },
    "llama3.1-70b": {
        "engine": "megatron",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_70b",
    },
    "llama3.1-405b": {
        "engine": "megatron",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_405b",
    },

    # qwen
    "qwen-1.8b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_1_8b",
    },
    "qwen-7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_7b",
    },
    "qwen-14b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_14b",
    },
    "qwen-72b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_72b",
    },
    "qwen1.5-0.5b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_0_5b",
    },
    "qwen1.5-1.8b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_1_8b",
    },
    "qwen1.5-4b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_4b",
    },
    "qwen1.5-7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_7b",
    },
    "qwen1.5-14b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_14b",
    },
    "qwen1.5-32b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_32b",
    },
    "qwen1.5-72b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_72b",
    },
    "qwen2-0.5b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_0_5b",
    },
    "qwen2-1.5b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_1_5b",
    },
    "qwen2-7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_7b",
    },
    "qwen2-72b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_72b",
    },
    "qwen2.5-0.5b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_0_5b",
    },
    "qwen2.5-1.5b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_1_5b",
    },
    "qwen2.5-3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_3b",
    },
    "qwen2.5-7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_7b",
    },
    "qwen2.5-14b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_14b",
    },
    "qwen2.5-32b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_32b",
    },
    "qwen2.5-72b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_72b",
    },
    "qwen3-0.6b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_0_6b",
    },
    "qwen3-1.7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_1_7b",
    },
    "qwen3-4b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_4b",
    },
    "qwen3-8b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_8b",
    },
    "qwen3-14b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_14b",
    },
    "qwen3-30b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_30b_a3b",
    },
    "qwen3-32b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_32b",
    },
    "qwen3-235b-a22b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_235b_a22b",
    },
    "qwen3-480b-a35b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_480b_a35b",
    },
    "qwen3-coder-30b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_coder_30b_a3b",
    },

    # qwen3-next-80b-a3b
    "qwen3-next-80b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3_next",
        "config_name": "qwen3_next_80b_a3b",
    },

    # Kimi K3 multimodal model
    "kimi-k3": {
        "engine": "megatron",
        "config_path": "configs/models/kimi_k3",
        "config_name": "kimi_k3",
    },

    # qwen3.5
    "qwen3.5-0.8b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_0_8b",
    },
    "qwen3.5-2b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_2b",
    },
    "qwen3.5-4b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_4b",
    },
    "qwen3.5-9b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_9b",
    },
    "qwen3.5-27b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_27b",
    },
    "qwen3.5-35b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_35b_a3b",
    },
    "qwen3.5-122b-a10b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_122b_a10b",
    },
    "qwen3.5-397b-a17b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_397b_a17b",
    },

    # qwen3.6
    "qwen3.6-27b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_27b",
    },
    "qwen3.6-35b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_35b_a3b",
    },

    # qwen3.8
    "qwen3.8-27b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3.8",
        "config_name": "qwen3_8_27b",
    },

    # kimi-k2.x
    "kimi-k2.5": {
        "engine": "megatron",
        "config_path": "configs/models/kimi_k2.5",
        "config_name": "kimi_k2_5",
    },
    "kimi-k2.6": {
        "engine": "megatron",
        "config_path": "configs/models/kimi_k2.6",
        "config_name": "kimi_k2_6",
    },
    "kimi-k2.6-bf16": {
        "engine": "megatron",
        "config_path": "configs/models/kimi_k2.6",
        "config_name": "kimi_k2_6_bf16",
    },

    # qwen2.5-vl
    "qwen2.5-vl-3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5_vl",
        "config_name": "qwen2_5_vl_3b",
    },
    "qwen2.5-vl-3b-lora": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5_vl",
        "config_name": "qwen2_5_vl_3b_lora",
    },
    "qwen2.5-vl-7b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5_vl",
        "config_name": "qwen2_5_vl_7b",
    },
    "qwen2.5-vl-32b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5_vl",
        "config_name": "qwen2_5_vl_32b",
    },
    "qwen2.5-vl-72b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen2.5_vl",
        "config_name": "qwen2_5_vl_72b",
    },

    # internvl 2.5
    "internvl2.5-8b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_8b",
    },
    "internvl2.5-26b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_26b",
    },
    "internvl2.5-38b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_38b",
    },
    "internvl2.5-78b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_78b",
    },

    # internvl 3.5
    "internvl3.5-8b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_8b",
    },
    "internvl3.5-14b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_14b",
    },
    "internvl3.5-30b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_30b_a3b",
    },
    "internvl3.5-38b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_38b",
    },
    "internvl3.5-241b-a28b": {
        "engine": "megatron",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_241b_a28b",
    },

    # llavaov 1.5
    "llava-onevision-1.5-4b": {
        "engine": "megatron",
        "config_path": "configs/models/llava_onevision_1.5",
        "config_name": "llava_onevision_1_5_4b",
    },

    # qwen3-vl
    "qwen3-vl-30b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_30b_a3b",
    },
    "qwen3-vl-235b-a22b": {
        "engine": "megatron",
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_235b_a22b",
    },

    # minicpm-v
    "minicpm-v-4.6": {
        "engine": "megatron",
        "config_path": "configs/models/minicpm_v_4_6",
        "config_name": "minicpm_v_4_6",
    },

    # wan
    "wan2-1-i2v": {
        "engine": "megatron",
        "config_path": "configs/models/wan",
        "config_name": "wan2_1_i2v",
    },
    "wan2-2-i2v": {
        "engine": "megatron",
        "config_path": "configs/models/wan",
        "config_name": "wan2_2_i2v",
    },

    # qwen image
    "qwen-image-edit-2511": {
        "engine": "megatron",
        "config_path": "configs/models/qwen_image",
        "config_name": "qwen_image_edit_2511",
    },

    # mimo
    "mimo": {
        "engine": "megatron",
        "config_path": "configs/models/mimo",
        "config_name": "mimo_7b",
    },

    # minimax
    "minimax2.1-230b": {
        "engine": "megatron",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_1",
    },
    "minimax2.5-230b": {
        "engine": "megatron",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_5",
    },
    "minimax2.7-230b": {
        "engine": "megatron",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_7",
    },

    # ernie4.5-vl
    "ernie4.5-28b-a3b-base": {
        "engine": "megatron",
        "config_path": "configs/models/ernie4.5",
        "config_name": "ernie4_5_28b_a3b_base",
    },
    "ernie4.5-vl-28b-a3b": {
        "engine": "megatron",
        "config_path": "configs/models/ernie4.5",
        "config_name": "ernie4_5_vl_28b_a3b",
    },
    "glm5": {
        "engine": "megatron",
        "config_path": "configs/models/glm5",
        "config_name": "glm5",
    },
    "glm5.2": {
        "engine": "megatron",
        "config_path": "configs/models/glm5.2",
        "config_name": "glm5_2",
    },
    "glm5.3-flash": {
        "engine": "megatron",
        "config_path": "configs/models/glm5_next",
        "config_name": "glm5.3_flash",
    },
    "lingbot_va_robotwin": {
        "engine": "torch",
        "config_path": "configs/models/lingbot_va",
        "config_name": "lingbot_va_robotwin",
        "model_config": "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
        "data_config": "loongforge.data.embodied.lingbot_va.transforms.data_configuration_lingbot_va.LingBotVADataConfig",
    },
    "lingbot_va_libero": {
        "engine": "torch",
        "config_path": "configs/models/lingbot_va",
        "config_name": "lingbot_va_libero",
        "model_config": "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
        "data_config": "loongforge.data.embodied.lingbot_va.transforms.data_configuration_lingbot_va.LingBotVADataConfig",
    },
    "pi05": {
        "engine": "torch",
        "config_path": "configs/models/pi05",
        "config_name": "pi05",
        "model_config": "loongforge.models.embodied.pi05.model_configuration_pi05.Pi05ModelConfig",
        "data_config": "loongforge.data.embodied.pi05.transforms.data_configuration_pi05.Pi05DataConfig",
    },
    "gr00t_n1_6": {
        "engine": "torch",
        "config_path": "configs/models/gr00t_n1_6",
        "config_name": "gr00t_n1_6",
        "model_config": "loongforge.models.embodied.gr00t_n1_6.model_configuration_gr00t_n1_6.Gr00tN1d6ModelConfig",
        "data_config": "loongforge.data.embodied.gr00t_n1_6.transforms.data_configuration_gr00t_n1_6.Gr00tN1d6DataConfig",
    },
    "xvla": {
        "engine": "torch",
        "config_path": "configs/models/xvla",
        "config_name": "xvla",
        "model_config": "loongforge.models.embodied.xvla.model_configuration_xvla.XVLAModelConfig",
        "data_config": "loongforge.data.embodied.xvla.transforms.data_configuration_xvla.XVLADataConfig",
    },
    "vlda": {
        "engine": "torch",
        "config_path": "configs/models/vlda",
        "config_name": "vlda",
        "model_config": "loongforge.models.embodied.vlda.model_configuration_vlda.VLDAModelConfig",
        "data_config": "loongforge.data.embodied.vlda.transforms.data_configuration_vlda.VLDADataConfig",
    },
    "fastwam": {
        "engine": "torch",
        "config_path": "configs/models/fastwam",
        "config_name": "fastwam",
        "model_config": "loongforge.models.embodied.fastwam.modeling_configuration_fastwam.FastWAMModelConfig",
        "data_config": "loongforge.data.embodied.fastwam.transforms.data_configuration_fastwam.FastWAMDataConfig",
    },
    "gr00t_n1_7": {
        "engine": "torch",
        "config_path": "configs/models/gr00t_n1_7",
        "config_name": "gr00t_n1_7",
        "model_config": "loongforge.models.embodied.gr00t_n1_7.model_configuration_gr00t_n1_7.Gr00tN1d7Config",
        "data_config": "loongforge.data.embodied.gr00t_n1_7.transforms.data_configuration_gr00t_n1_7.Gr00tN1d7DataConfig",
    },
    "cosmos3_nano": {
        "engine": "torch",
        "config_path": "configs/models/cosmos3",
        "config_name": "nano",
        "model_config": "loongforge.models.embodied.cosmos3.modeling_configuration_cosmos3.Cosmos3ModelConfig",
        "data_config": "loongforge.data.embodied.cosmos3.data_configuration_cosmos3.Cosmos3DataConfig",
    },
    "dreamzero_lora_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_full_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_lora_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_full_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_libero_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_libero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_agibot_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_agibot_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_yam_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/dreamzero",
        "config_name": "dreamzero_yam_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "wall_oss_0_5": {
        "engine": "torch",
        "config_path": "configs/models/wall_oss_0_5",
        "config_name": "wall_oss_0_5",
        "model_config": "loongforge.models.embodied.wall_oss_0_5.model_configuration_wall_oss_0_5.WallOss05ModelConfig",
        "data_config": "loongforge.data.embodied.wall_oss_0_5.transforms.data_configuration_wall_oss_0_5.WallOss05DataConfig",
    },
}


def get_model_entry(model_name: str):
    """Resolve a name while preserving Torch's hyphen aliases."""
    name = model_name.lower()
    names = {name, name.replace("-", "_")}
    matches = [
        entry for key in names
        if (entry := MODEL_CONFIG_REGISTRY.get(key)) is not None
        and (key == name or entry["engine"] == "torch")
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous model name: {model_name!r}")
    if not matches:
        raise ValueError(f"Unknown model name: {model_name!r}. Register it in loongforge/models/catalog.py.")
    return matches[0]


def get_config_from_model_name(model_name: str):
    """Return the absolute YAML directory and config name."""
    entry = get_model_entry(model_name)
    root = Path(__file__).resolve().parents[2]
    return str(root / entry["config_path"]), entry["config_name"]


def get_config_path(model_name: str):
    """Return an existing default YAML path."""
    directory, name = get_config_from_model_name(model_name)
    path = Path(directory) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return str(path)


def get_config_types(model_name: str):
    """Load only the selected Torch model's config types."""
    entry = get_model_entry(model_name)
    if entry["engine"] != "torch":
        raise ValueError(f"Model {model_name!r} does not use Torch config types")
    classes = []
    for field in ("model_config", "data_config"):
        module, name = entry[field].rsplit(".", 1)
        classes.append(getattr(import_module(module), name))
    return tuple(classes)


######## built-in models #######
# Using List[str] instead of list[str] to ensure compatibility with older versions of Python(<3.9)
class _BaseFamilies(object):
    @classmethod
    def names(cls) -> List[str]:
        """Return a list of all string names defined in the class and its subclasses"""
        string_names = [
            value
            for name, value in vars(cls).items()
            if isinstance(value, str) and not name.startswith("__")
        ]
        return string_names


class LanguageModelFamilies(_BaseFamilies):
    """Language model families"""

    LLAMA = "llama"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    LLAMA3_1 = "llama3.1"
    QWEN = "qwen"
    QWEN1_5 = "qwen1.5"
    QWEN2 = "qwen2"
    QWEN2_5 = "qwen2.5"
    QWEN3 = "qwen3"
    QWEN3_NEXT = "qwen3_next"
    KIMI_K3_LLM = "kimi_k3_llm"
    DEEPSEEK = "deepseek"
    DEEPSEEK_V4 = "deepseek_v4"
    INTERNLM2_5 = "internlm2.5"
    MINIMAX = "minimax"
    MIMO = "mimo"
    GLM = "glm"
    GLM5_NEXT = "glm5_next"


class VisionLanguageModelFamilies(_BaseFamilies):
    """Vision language model families"""

    QWEN2_VL = "qwen2_vl"
    QWEN2_5_VL = "qwen2_5_vl"
    QWEN3_VL = "qwen3_vl"
    LLAVA_OV_1_5 = "llava_ov_1_5"
    VLM = "vlm"
    INTERN_VL = "intern_vl"
    ERNIE4_5_VL = "ernie4_5_vl"
    QWEN3_5 = "qwen3_5"
    MINICPM_V_4_6 = "minicpm_v_4_6"
    KIMI_K2_5 = "kimi_k2_5"
    KIMI_K2_6 = "kimi_k2_6"
    KIMI_K3 = "kimi_k3"


class DiffusionModelFamilies(_BaseFamilies):
    """Diffusion model families (Megatron engine)"""

    WAN2_1_I2V = "wan2_1_i2v"
    WAN2_2_I2V = "wan2_2_i2v"
    QWEN_IMAGE = "qwen_image"


# Families whose Hydra config puts model_type at the top level; VLM configs put it under ``model``.
TOP_LEVEL_MODEL_FAMILIES = frozenset(LanguageModelFamilies.names() + DiffusionModelFamilies.names())


def split_model_config(config):
    """Return ``(model_type, model_config)`` of a composed Megatron Hydra config."""
    if hasattr(config, "model_type") and config.model_type in TOP_LEVEL_MODEL_FAMILIES:
        return config.model_type, config
    if not hasattr(config, "model"):
        raise ValueError("Invalid model configuration structure")
    return config.model.model_type, config.model


def get_all_model_families() -> List[str]:
    """
    Get all model families defined in this module dynamically.
    Returns a flattened list of all string names from all subclasses of _BaseFamilies.
    """
    all_families = []

    for family_class in _BaseFamilies.__subclasses__():
        all_families.extend(family_class.names())

    return all_families
