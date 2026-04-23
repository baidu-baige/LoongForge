# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""config mapping"""

from pathlib import Path


# registry for model config
MODEL_CONFIG_REGISTRY = {
    # deepseek
    "deepseek-v2": {
        "config_path": "configs/models/deepseek2",
        "config_name": "deepseek_v2",
    },
    "deepseek-v2-lite": {
        "config_path": "configs/models/deepseek2",
        "config_name": "deepseek_v2_lite",
    },
    "deepseek-v3": {
        "config_path": "configs/models/deepseek3",
        "config_name": "deepseek_v3",
    },
    "deepseek-v3.2-sparse": {
        "config_path": "configs/models/deepseek3",
        "config_name": "deepseek_v3_2_sparse",
    },
    "deepseek-v3.2-warmup": {
        "config_path": "configs/models/deepseek3",
        "config_name": "deepseek_v3_2_warmup",
    },
    # internlm2.5
    "internlm2.5-8b": {
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_8b",
    },
    "internlm2.5-20b": {
        "config_path": "configs/models/internlm2.5",
        "config_name": "internlm2_5_20b",
    },
    # llama
    "llama2-7b": {
        "config_path": "configs/models/llama2",
        "config_name": "llama2_7b",
    },
    "llama2-13b": {
        "config_path": "configs/models/llama2",
        "config_name": "llama2_13b",
    },
    "llama2-70b": {
        "config_path": "configs/models/llama2",
        "config_name": "llama2_70b",
    },
    "llama3-8b": {
        "config_path": "configs/models/llama3",
        "config_name": "llama3_8b",
    },
    "llama3-70b": {
        "config_path": "configs/models/llama3",
        "config_name": "llama3_70b",
    },
    "llama3.1-8b": {
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_8b",
    },
    "llama3.1-70b": {
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_70b",
    },
    "llama3.1-405b": {
        "config_path": "configs/models/llama3",
        "config_name": "llama3_1_405b",
    },

    # qwen
    "qwen-1.8b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen_1_8b",
    },
    "qwen-7b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen_7b",
    },
    "qwen-14b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen_14b",
    },
    "qwen-72b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen_72b",
    },
    "qwen1.5-0.5b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_0_5b",
    },
    "qwen1.5-1.8b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_1_8b",
    },
    "qwen1.5-4b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_4b",
    },
    "qwen1.5-7b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_7b",
    },
    "qwen1.5-14b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_14b",
    },
    "qwen1.5-32b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_32b",
    },
    "qwen1.5-72b": {
        "config_path": "configs/models/qwen",
        "config_name": "qwen1_5_72b",
    },
    "qwen2-0.5b": {
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_0_5b",
    },
    "qwen2-1.5b": {
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_1_5b",
    },
    "qwen2-7b": {
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_7b",
    },
    "qwen2-72b": {
        "config_path": "configs/models/qwen2",
        "config_name": "qwen2_72b",
    },
    "qwen2.5-0.5b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_0_5b",
    },
    "qwen2.5-1.5b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_1_5b",
    },
    "qwen2.5-3b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_3b",
    },
    "qwen2.5-7b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_7b",
    },
    "qwen2.5-14b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_14b",
    },
    "qwen2.5-32b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_32b",
    },
    "qwen2.5-72b": {
        "config_path": "configs/models/qwen2.5",
        "config_name": "qwen2_5_72b",
    },
    "qwen3-0.6b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_0_6b",
    },
    "qwen3-1.7b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_1_7b",
    },
    "qwen3-4b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_4b",
    },
    "qwen3-8b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_8b",
    },
    "qwen3-14b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_14b",
    },
    "qwen3-30b-a3b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_30b_a3b",
    },
    "qwen3-32b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_32b",
    },
    "qwen3-235b-a22b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_235b_a22b",
    },
    "qwen3-480b-a35b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_480b_a35b",
    },
    "qwen3-coder-30b-a3b": {
        "config_path": "configs/models/qwen3",
        "config_name": "qwen3_coder_30b_a3b",
    },

    # qwen3-next-80b-a3b
    "qwen3-next-80b-a3b": {
        "config_path": "configs/models/qwen3_next",
        "config_name": "qwen3_next_80b_a3b",
    },
    
    # qwen3.5
    "qwen3.5-0.8b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_0_8b",
    },
    "qwen3.5-2b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_2b",
    },
    "qwen3.5-4b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_4b",
    },
    "qwen3.5-9b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_9b",
    },
    "qwen3.5-27b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_27b",
    },
    "qwen3.5-35b-a3b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_35b_a3b",
    },
    "qwen3.5-122b-a10b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_122b_a10b",
    },
    "qwen3.5-397b-a17b": {
        "config_path": "configs/models/qwen3.5",
        "config_name": "qwen3_5_397b_a17b",
    },

    # qwen3.6
    "qwen3.6-27b": {
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_27b",
    },
    "qwen3.6-35b-a3b": {
        "config_path": "configs/models/qwen3.6",
        "config_name": "qwen3_6_35b_a3b",
    },
    
    # qwen2.5-vl
    "qwen2.5-vl-3b": {
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_3b",
    },
    "qwen2.5-vl-3b-lora": {
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_3b_lora",
    },
    "qwen2.5-vl-7b": {
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_7b",
    },
    "qwen2.5-vl-32b": {
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_32b",
    },
    "qwen2.5-vl-72b": {
        "config_path": "configs/models/qwen2.5vl",
        "config_name": "qwen2_5_vl_72b",
    },

    # internvl 2.5
    "internvl2.5-8b": {
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_8b",
    },
    "internvl2.5-26b": {
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_26b",
    },
    "internvl2.5-38b": {
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_38b",
    },
    "internvl2.5-78b": {
        "config_path": "configs/models/internvl2.5",
        "config_name": "internvl2_5_78b",
    },

    # internvl 3.5
    "internvl3.5-8b": {
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_8b",
    },
    "internvl3.5-14b": {
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_14b",
    },
    "internvl3.5-30b-a3b": {
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_30b_a3b",
    },
    "internvl3.5-38b": {
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_38b",
    },
    "internvl3.5-241b-a28b": {
        "config_path": "configs/models/internvl3.5",
        "config_name": "internvl3_5_241b_a28b",
    },

    # llavaov 1.5
    "llava-onevision-1.5-4b": {
        "config_path": "configs/models/llava_onevision",
        "config_name": "llava_onevision_1_5_4b",
    },

    # qwen3-vl
    "qwen3-vl-30b-a3b": {
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_30b_a3b",
    },
    "qwen3-vl-235b-a22b": {
        "config_path": "configs/models/qwen3_vl",
        "config_name": "qwen3_vl_235b_a22b",
    },

    # wan2.2
    "wan2-2-i2v": {
        "config_path": "configs/models/wan",
        "config_name": "wan2_2_i2v",
    },

    # pi05
    "pi05": {
        # Hydra expects the directory, config name selects the file.
        "config_path": "configs/models/pi05",
        "config_name": "pi05",
    },

    # groot
    "groot_n1_6": {
        "config_path": "configs/models/groot",
        "config_name": "groot_n1_6",
    },

    # mimo
    "mimo": {
        "config_path": "configs/models/mimo",
        "config_name": "mimo_7b",
    },

    # minimax
    "minimax2.1-230b": {
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_1",
    },
    "minimax2.5-230b": {
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_5",
    },

    # ernie4.5-vl
    "ernie4.5-28b-a3b-base": {
        "config_path": "configs/models/ernie4.5vl",
        "config_name": "ernie4_5_28b_a3b_base",
    },
    "ernie4.5-vl-28b-a3b": {
        "config_path": "configs/models/ernie4.5vl",
        "config_name": "ernie4_5_vl_28b_a3b",
    },
    "llava-onevision-1.5-4b": {
        "config_path": "configs/models/llava_onevision",
        "config_name": "llava_onevision_1_5_4b",
    },
    "glm5": {
        "config_path": "configs/models/glm5",
        "config_name": "glm5",
    },
}


def normalize_model_name(name: str) -> str:
    """Normalize input model name into canonical form."""
    # in case of adding other regular expressions
    return name.lower()


def get_config_from_model_name(model_name: str):
    """
    Lookup (config_path, config_name) from MODEL_CONFIG_REGISTRY,
    and automatically prepend absolute project_root to config_path.
    """
    name = normalize_model_name(model_name)

    if name not in MODEL_CONFIG_REGISTRY:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"You may consider passing the --config-file directly or"
            f"register the config file path in config_map.py to add the model name '{model_name}'."
            f"Now the available model names are: {list(MODEL_CONFIG_REGISTRY.keys())}"
        )
    entry = MODEL_CONFIG_REGISTRY[name]

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent   # LoongForge root dir

    abs_config_path = str(project_root / entry["config_path"])

    return abs_config_path, entry["config_name"]
