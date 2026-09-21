# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model names, engines, default YAMLs, and lazy config type bindings."""

from importlib import import_module
from pathlib import Path


MODEL_CONFIG_REGISTRY = {
    # deepseek
    "deepseek-v2": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek2",
        "config_name": "deepseek_v2",
    },
    "deepseek-v2-lite": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek2",
        "config_name": "deepseek_v2_lite",
    },
    "deepseek-v3": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek3",
        "config_name": "deepseek_v3",
    },
    "deepseek-v3.2-sparse": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek3",
        "config_name": "deepseek_v3_2_sparse",
    },
    "deepseek-v4-flash": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_base",
    },
    "deepseek-v4-flash-lite": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite",
    },
    "deepseek-v4-flash-lite-2l": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_2l",
    },
    "deepseek-v4-flash-lite-4l": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_4l",
    },
    "deepseek-v4-flash-lite-6l": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_6l",
    },
    "deepseek-v4-pro": {
        "engine": "mcore",
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_pro_base",
    },
    # internlm2.5
    "internlm2.5-8b": {
        "engine": "mcore",
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_8b",
    },
    "internlm2.5-20b": {
        "engine": "mcore",
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_20b",
    },
    # llama
    "llama2-7b": {
        "engine": "mcore",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_7b",
    },
    "llama2-13b": {
        "engine": "mcore",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_13b",
    },
    "llama2-70b": {
        "engine": "mcore",
        "config_path": "configs/models/llama2",
        "config_name": "llama2_70b",
    },
    "llama3-8b": {
        "engine": "mcore",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_8b",
    },
    "llama3-70b": {
        "engine": "mcore",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_70b",
    },
    "llama3.1-8b": {
        "engine": "mcore",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_8b",
    },
    "llama3.1-70b": {
        "engine": "mcore",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_70b",
    },
    "llama3.1-405b": {
        "engine": "mcore",
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_405b",
    },

    # qwen
    "qwen-1.8b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_1_8b",
    },
    "qwen-7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_7b",
    },
    "qwen-14b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_14b",
    },
    "qwen-72b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen_72b",
    },
    "qwen1.5-0.5b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_0_5b",
    },
    "qwen1.5-1.8b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_1_8b",
    },
    "qwen1.5-4b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_4b",
    },
    "qwen1.5-7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_7b",
    },
    "qwen1.5-14b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_14b",
    },
    "qwen1.5-32b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_32b",
    },
    "qwen1.5-72b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_72b",
    },
    "qwen2-0.5b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_0_5b",
    },
    "qwen2-1.5b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_1_5b",
    },
    "qwen2-7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_7b",
    },
    "qwen2-72b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_72b",
    },
    "qwen2.5-0.5b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_0_5b",
    },
    "qwen2.5-1.5b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_1_5b",
    },
    "qwen2.5-3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_3b",
    },
    "qwen2.5-7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_7b",
    },
    "qwen2.5-14b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_14b",
    },
    "qwen2.5-32b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_32b",
    },
    "qwen2.5-72b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_72b",
    },
    "qwen3-0.6b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_0_6b",
    },
    "qwen3-1.7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_1_7b",
    },
    "qwen3-4b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_4b",
    },
    "qwen3-8b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_8b",
    },
    "qwen3-14b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_14b",
    },
    "qwen3-30b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_30b_a3b",
    },
    "qwen3-32b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_32b",
    },
    "qwen3-235b-a22b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_235b_a22b",
    },
    "qwen3-480b-a35b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_480b_a35b",
    },
    "qwen3-coder-30b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_coder_30b_a3b",
    },

    # qwen3-next-80b-a3b
    "qwen3-next-80b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3_next",
        "config_name": "qwen3_next_80b_a3b",
    },

    # Kimi K3 multimodal model
    "kimi-k3": {
        "engine": "mcore",
        "config_path": "configs/models/kimi_k3",
        "config_name": "kimi_k3",
    },

    # qwen3.5
    "qwen3.5-0.8b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_0_8b",
    },
    "qwen3.5-2b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_2b",
    },
    "qwen3.5-4b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_4b",
    },
    "qwen3.5-9b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_9b",
    },
    "qwen3.5-27b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_27b",
    },
    "qwen3.5-35b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_35b_a3b",
    },
    "qwen3.5-122b-a10b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_122b_a10b",
    },
    "qwen3.5-397b-a17b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_397b_a17b",
    },

    # qwen3.6
    "qwen3.6-27b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_27b",
    },
    "qwen3.6-35b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_35b_a3b",
    },

    # qwen3.8
    "qwen3.8-27b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3.8",
        "config_name": "qwen3_8_27b",
    },

    # kimi-k2.x
    "kimi-k2.5": {
        "engine": "mcore",
        "config_path": "configs/models/kimi_k2.5",
        "config_name": "kimi_k2_5",
    },
    "kimi-k2.6": {
        "engine": "mcore",
        "config_path": "configs/models/kimi_k2.6",
        "config_name": "kimi_k2_6",
    },

    # qwen2.5-vl
    "qwen2.5-vl-3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_3b",
    },
    "qwen2.5-vl-3b-lora": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_3b_lora",
    },
    "qwen2.5-vl-7b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_7b",
    },
    "qwen2.5-vl-32b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_32b",
    },
    "qwen2.5-vl-72b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_72b",
    },

    # internvl 2.5
    "internvl2.5-8b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_8b",
    },
    "internvl2.5-26b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_26b",
    },
    "internvl2.5-38b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_38b",
    },
    "internvl2.5-78b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_78b",
    },

    # internvl 3.5
    "internvl3.5-8b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_8b",
    },
    "internvl3.5-14b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_14b",
    },
    "internvl3.5-30b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_30b_a3b",
    },
    "internvl3.5-38b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_38b",
    },
    "internvl3.5-241b-a28b": {
        "engine": "mcore",
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_241b_a28b",
    },

    # llavaov 1.5
    "llava-onevision-1.5-4b": {
        "engine": "mcore",
        "config_path": "configs/models/llava_onevision",
        "config_name": "llava_onevision_1_5_4b",
    },

    # qwen3-vl
    "qwen3-vl-30b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_30b_a3b",
    },
    "qwen3-vl-235b-a22b": {
        "engine": "mcore",
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_235b_a22b",
    },

    # minicpm-v
    "minicpm-v-4.6": {
        "engine": "mcore",
        "config_path": "configs/models/minicpm_v_4_6",
        "config_name": "minicpm_v_4_6",
    },

    # wan
    "wan2-1-i2v": {
        "engine": "mcore",
        "config_path": "configs/models/wan",
        "config_name": "wan2_1_i2v",
    },
    "wan2-2-i2v": {
        "engine": "mcore",
        "config_path": "configs/models/wan",
        "config_name": "wan2_2_i2v",
    },

    # qwen image
    "qwen-image-edit-2511": {
        "engine": "mcore",
        "config_path": "configs/models/qwen_image",
        "config_name": "qwen_image_edit_2511",
    },

    # mimo
    "mimo": {
        "engine": "mcore",
        "config_path": "configs/models/mimo",
        "config_name": "mimo_7b",
    },

    # minimax
    "minimax2.1-230b": {
        "engine": "mcore",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_1",
    },
    "minimax2.5-230b": {
        "engine": "mcore",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_5",
    },
    "minimax2.7-230b": {
        "engine": "mcore",
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_7",
    },

    # ernie4.5-vl
    "ernie4.5-28b-a3b-base": {
        "engine": "mcore",
        "config_path": "configs/models/ernie4_5_vl",
        "config_name": "ernie4_5_28b_a3b_base",
    },
    "ernie4.5-vl-28b-a3b": {
        "engine": "mcore",
        "config_path": "configs/models/ernie4_5_vl",
        "config_name": "ernie4_5_vl_28b_a3b",
    },
    "glm5": {
        "engine": "mcore",
        "config_path": "configs/models/glm5",
        "config_name": "glm5",
    },
    "glm5.2": {
        "engine": "mcore",
        "config_path": "configs/models/glm5.2",
        "config_name": "glm5_2",
    },
    "lingbot_va_robotwin": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "lingbot_va_robotwin",
        "model_config": "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
        "data_config": "loongforge.data.embodied.datasets.lingbot_va.transforms.data_configuration_lingbot_va.LingBotVADataConfig",
    },
    "lingbot_va_libero": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "lingbot_va_libero",
        "model_config": "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
        "data_config": "loongforge.data.embodied.datasets.lingbot_va.transforms.data_configuration_lingbot_va.LingBotVADataConfig",
    },
    "pi05": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "pi05",
        "model_config": "loongforge.models.embodied.pi05.model_configuration_pi05.Pi05ModelConfig",
        "data_config": "loongforge.data.embodied.datasets.pi05.transforms.data_configuration_pi05.Pi05DataConfig",
    },
    "groot_n1_6": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "groot_n1_6",
        "model_config": "loongforge.models.embodied.groot_n1_6.model_configuration_groot_n1_6.GrootN1d6ModelConfig",
        "data_config": "loongforge.data.embodied.datasets.groot_n1_6.transforms.data_configuration_groot_n1_6.GrootN1d6DataConfig",
    },
    "xvla": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "xvla",
        "model_config": "loongforge.models.embodied.xvla.model_configuration_xvla.XvlaModelConfig",
        "data_config": "loongforge.data.embodied.datasets.xvla.transforms.data_configuration_xvla.XvlaDataConfig",
    },
    "fastwam": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "fastwam",
        "model_config": "loongforge.models.embodied.fastwam.modeling_configuration_fastwam.FastWAMModelConfig",
        "data_config": "loongforge.data.embodied.datasets.fastwam.transforms.data_configuration_fastwam.FastWAMDataConfig",
    },
    "groot_n1_7": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "groot_n1_7",
        "model_config": "loongforge.models.embodied.groot_n1_7.model_configuration_groot_n1_7.GrootN1d7Config",
        "data_config": "loongforge.data.embodied.datasets.groot_n1_7.transforms.data_configuration_groot_n1_7.GrootN1d7DataConfig",
    },
    "cosmos3_nano": {
        "engine": "torch",
        "config_path": "configs/models/embodied/cosmos3",
        "config_name": "nano",
        "model_config": "loongforge.models.embodied.cosmos3.modeling_configuration_cosmos3.Cosmos3ModelConfig",
        "data_config": "loongforge.data.embodied.datasets.cosmos3.data_configuration_cosmos3.Cosmos3DroidConfig",
    },
    "dreamzero_lora_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_full_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_lora_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_full_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_libero_wan22_5b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_libero_wan22_5b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_agibot_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_agibot_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "dreamzero_yam_wan21_14b": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "dreamzero_yam_wan21_14b",
        "model_config": "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
        "data_config": "loongforge.data.embodied.datasets.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
    },
    "wall_oss_0_5": {
        "engine": "torch",
        "config_path": "configs/models/embodied",
        "config_name": "wall_oss_0_5",
        "model_config": "loongforge.models.embodied.wall_oss_0_5.model_configuration_wall_oss_0_5.WallOss05ModelConfig",
        "data_config": "loongforge.data.embodied.datasets.wall_oss_0_5.transforms.data_configuration_wall_oss_0_5.WallOss05DataConfig",
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
