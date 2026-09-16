# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model catalog without training or accelerator imports."""

from pathlib import Path

from loongforge.contracts.model import ModelSpec


# registry for model config
MCORE_CONFIGS = {
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
    "deepseek-v4-flash": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_base",
    },
    "deepseek-v4-flash-lite": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite",
    },
    "deepseek-v4-flash-lite-2l": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_2l",
    },
    "deepseek-v4-flash-lite-4l": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_4l",
    },
    "deepseek-v4-flash-lite-6l": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_flash_lite_6l",
    },
    "deepseek-v4-pro": {
        "config_path": "configs/models/deepseek4",
        "config_name": "deepseek_v4_pro_base",
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

    # Kimi K3 multimodal model
    "kimi-k3": {
        "config_path": "configs/models/kimi_k3",
        "config_name": "kimi_k3",
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

    # qwen3.8
    "qwen3.8-27b": {
        "config_path": "configs/models/qwen3.8",
        "config_name": "qwen3_8_27b",
    },

    # kimi-k2.x
    "kimi-k2.5": {
        "config_path": "configs/models/kimi_k2.5",
        "config_name": "kimi_k2_5",
    },
    "kimi-k2.6": {
        "config_path": "configs/models/kimi_k2.6",
        "config_name": "kimi_k2_6",
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

    # minicpm-v
    "minicpm-v-4.6": {
        "config_path": "configs/models/minicpm_v_4_6",
        "config_name": "minicpm_v_4_6",
    },

    # wan
    "wan2-1-i2v": {
        "config_path": "configs/models/wan",
        "config_name": "wan2_1_i2v",
    },
    "wan2-2-i2v": {
        "config_path": "configs/models/wan",
        "config_name": "wan2_2_i2v",
    },

    # qwen image
    "qwen-image-edit-2511": {
        "config_path": "configs/models/qwen_image",
        "config_name": "qwen_image_edit_2511",
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
    "minimax2.7-230b": {
        "config_path": "configs/models/minimax",
        "config_name": "minimax_m2_7",
    },

    # ernie4.5-vl
    "ernie4.5-28b-a3b-base": {
        "config_path": "configs/models/ernie4_5_vl",
        "config_name": "ernie4_5_28b_a3b_base",
    },
    "ernie4.5-vl-28b-a3b": {
        "config_path": "configs/models/ernie4_5_vl",
        "config_name": "ernie4_5_vl_28b_a3b",
    },
    "glm5": {
        "config_path": "configs/models/glm5",
        "config_name": "glm5",
    },
    "glm5.2": {
        "config_path": "configs/models/glm5.2",
        "config_name": "glm5_2",
    },
}


_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"

def _torch(yaml_file, model, data):
    return ModelSpec("torch", _CONFIGS_DIR / yaml_file, model, data)


_PI05 = (
    "loongforge.models.vla.pi05.model_configuration_pi05:Pi05ModelConfig",
    "loongforge.datasets.robotics.pi05.transforms.data_configuration_pi05:Pi05DataConfig",
)
_GROOT16 = (
    "loongforge.models.vla.groot_n1_6.model_configuration_groot_n1_6:GrootN1d6ModelConfig",
    "loongforge.datasets.robotics.groot_n1_6.transforms.data_configuration_groot_n1_6:GrootN1d6DataConfig",
)
_GROOT17 = (
    "loongforge.models.vla.groot_n1_7.model_configuration_groot_n1_7:GrootN1d7Config",
    "loongforge.datasets.robotics.groot_n1_7.transforms.data_configuration_groot_n1_7:GrootN1d7DataConfig",
)
_XVLA = (
    "loongforge.models.vla.xvla.model_configuration_xvla:XvlaModelConfig",
    "loongforge.datasets.robotics.xvla.transforms.data_configuration_xvla:XvlaDataConfig",
)
_FASTWAM = (
    "loongforge.models.world.fastwam.modeling_configuration_fastwam:FastWAMModelConfig",
    "loongforge.datasets.world.fastwam.transforms.data_configuration_fastwam:FastWAMDataConfig",
)
_COSMOS3 = (
    "loongforge.models.world.cosmos3.modeling_configuration_cosmos3:Cosmos3ModelConfig",
    "loongforge.datasets.world.cosmos3.data_configuration_cosmos3:Cosmos3DroidConfig",
)
_DREAMZERO = (
    "loongforge.models.world.dreamzero.model_configuration_dreamzero:DreamZeroConfig",
    "loongforge.datasets.world.dreamzero.transforms.data_configuration_dreamzero:DreamZeroDataConfig",
)
_LINGBOT = (
    "loongforge.models.world.lingbot_va.model_configuration_lingbot_va:LingBotVAModelConfig",
    "loongforge.datasets.world.lingbot_va.transforms.data_configuration_lingbot_va:LingBotVADataConfig",
)
_WALL = (
    "loongforge.models.vla.wall_oss_0_5.model_configuration_wall_oss_0_5:WallOss05ModelConfig",
    "loongforge.datasets.robotics.wall_oss_0_5.transforms.data_configuration_wall_oss_0_5:WallOss05DataConfig",
)


TORCH_CONFIGS = {
    "lingbot_va_robotwin": _torch("world/lingbot_va_robotwin.yaml", *_LINGBOT),
    "lingbot_va_libero": _torch("world/lingbot_va_libero.yaml", *_LINGBOT),
    "pi05": _torch("vla/pi05.yaml", *_PI05),
    "groot_n1_6": _torch("vla/groot_n1_6.yaml", *_GROOT16),
    "xvla": _torch("vla/xvla.yaml", *_XVLA),
    "fastwam": _torch("world/fastwam.yaml", *_FASTWAM),
    "groot_n1_7": _torch("vla/groot_n1_7.yaml", *_GROOT17),
    "cosmos3_nano": _torch("world/cosmos3/nano.yaml", *_COSMOS3),
    "dreamzero_lora_wan22_5b": _torch("world/dreamzero_wan22_5b.yaml", *_DREAMZERO),
    "dreamzero_full_wan22_5b": _torch("world/dreamzero_wan22_5b.yaml", *_DREAMZERO),
    "dreamzero_lora_wan21_14b": _torch("world/dreamzero_wan21_14b.yaml", *_DREAMZERO),
    "dreamzero_full_wan21_14b": _torch("world/dreamzero_wan21_14b.yaml", *_DREAMZERO),
    "dreamzero_libero_wan22_5b": _torch("world/dreamzero_libero_wan22_5b.yaml", *_DREAMZERO),
    "dreamzero_agibot_wan21_14b": _torch("world/dreamzero_agibot_wan21_14b.yaml", *_DREAMZERO),
    "dreamzero_yam_wan21_14b": _torch("world/dreamzero_yam_wan21_14b.yaml", *_DREAMZERO),
    "wall_oss_0_5": _torch("vla/wall_oss_0_5.yaml", *_WALL),
}



def get_model_spec(model_name: str, engine: str | None = None) -> ModelSpec:
    """Resolve a registered model and reject unsupported engine selection."""
    torch_name = model_name.lower().replace("-", "_")
    if torch_name in TORCH_CONFIGS:
        spec = TORCH_CONFIGS[torch_name]
    elif model_name.lower() in MCORE_CONFIGS:
        entry = MCORE_CONFIGS[model_name.lower()]
        path = Path(__file__).resolve().parents[2] / entry["config_path"]
        spec = ModelSpec("mcore", path / (entry["config_name"] + ".yaml"))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    if engine is not None and spec.engine != engine:
        raise ValueError(f"Model {model_name} supports {spec.engine}, not {engine}")
    if not spec.config_file.is_file():
        raise ValueError(f"Model config does not exist: {spec.config_file}")
    return spec
