# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model catalog without training or accelerator imports."""

from pathlib import Path

from loongforge.contracts import ModelSpec


_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"


def _mcore(yaml_file: str) -> ModelSpec:
    """Catalog entry for a model trained by the Megatron-backed engine."""
    return ModelSpec("mcore", _CONFIGS_DIR / yaml_file)


def _torch(yaml_file: str, model: str, data: str) -> ModelSpec:
    """Catalog entry for a model trained by the standalone Torch engine."""
    return ModelSpec("torch", _CONFIGS_DIR / yaml_file, model, data)


MCORE_CONFIGS = {
    # deepseek
    "deepseek-v2": _mcore("deepseek2/deepseek_v2.yaml"),
    "deepseek-v2-lite": _mcore("deepseek2/deepseek_v2_lite.yaml"),
    "deepseek-v3": _mcore("deepseek3/deepseek_v3.yaml"),
    "deepseek-v3.2-sparse": _mcore("deepseek3/deepseek_v3_2_sparse.yaml"),
    "deepseek-v4-flash": _mcore("deepseek4/deepseek_v4_flash_base.yaml"),
    "deepseek-v4-flash-lite": _mcore("deepseek4/deepseek_v4_flash_lite.yaml"),
    "deepseek-v4-flash-lite-2l": _mcore("deepseek4/deepseek_v4_flash_lite_2l.yaml"),
    "deepseek-v4-flash-lite-4l": _mcore("deepseek4/deepseek_v4_flash_lite_4l.yaml"),
    "deepseek-v4-flash-lite-6l": _mcore("deepseek4/deepseek_v4_flash_lite_6l.yaml"),
    "deepseek-v4-pro": _mcore("deepseek4/deepseek_v4_pro_base.yaml"),
    # internlm2.5
    "internlm2.5-8b": _mcore("internlm2.5/internlm2_5_8b.yaml"),
    "internlm2.5-20b": _mcore("internlm2.5/internlm2_5_20b.yaml"),
    # llama
    "llama2-7b": _mcore("llama2/llama2_7b.yaml"),
    "llama2-13b": _mcore("llama2/llama2_13b.yaml"),
    "llama2-70b": _mcore("llama2/llama2_70b.yaml"),
    "llama3-8b": _mcore("llama3/llama3_8b.yaml"),
    "llama3-70b": _mcore("llama3/llama3_70b.yaml"),
    "llama3.1-8b": _mcore("llama3/llama3_1_8b.yaml"),
    "llama3.1-70b": _mcore("llama3/llama3_1_70b.yaml"),
    "llama3.1-405b": _mcore("llama3/llama3_1_405b.yaml"),

    # qwen
    "qwen-1.8b": _mcore("qwen/qwen_1_8b.yaml"),
    "qwen-7b": _mcore("qwen/qwen_7b.yaml"),
    "qwen-14b": _mcore("qwen/qwen_14b.yaml"),
    "qwen-72b": _mcore("qwen/qwen_72b.yaml"),
    "qwen1.5-0.5b": _mcore("qwen/qwen1_5_0_5b.yaml"),
    "qwen1.5-1.8b": _mcore("qwen/qwen1_5_1_8b.yaml"),
    "qwen1.5-4b": _mcore("qwen/qwen1_5_4b.yaml"),
    "qwen1.5-7b": _mcore("qwen/qwen1_5_7b.yaml"),
    "qwen1.5-14b": _mcore("qwen/qwen1_5_14b.yaml"),
    "qwen1.5-32b": _mcore("qwen/qwen1_5_32b.yaml"),
    "qwen1.5-72b": _mcore("qwen/qwen1_5_72b.yaml"),
    "qwen2-0.5b": _mcore("qwen2/qwen2_0_5b.yaml"),
    "qwen2-1.5b": _mcore("qwen2/qwen2_1_5b.yaml"),
    "qwen2-7b": _mcore("qwen2/qwen2_7b.yaml"),
    "qwen2-72b": _mcore("qwen2/qwen2_72b.yaml"),
    "qwen2.5-0.5b": _mcore("qwen2.5/qwen2_5_0_5b.yaml"),
    "qwen2.5-1.5b": _mcore("qwen2.5/qwen2_5_1_5b.yaml"),
    "qwen2.5-3b": _mcore("qwen2.5/qwen2_5_3b.yaml"),
    "qwen2.5-7b": _mcore("qwen2.5/qwen2_5_7b.yaml"),
    "qwen2.5-14b": _mcore("qwen2.5/qwen2_5_14b.yaml"),
    "qwen2.5-32b": _mcore("qwen2.5/qwen2_5_32b.yaml"),
    "qwen2.5-72b": _mcore("qwen2.5/qwen2_5_72b.yaml"),
    "qwen3-0.6b": _mcore("qwen3/qwen3_0_6b.yaml"),
    "qwen3-1.7b": _mcore("qwen3/qwen3_1_7b.yaml"),
    "qwen3-4b": _mcore("qwen3/qwen3_4b.yaml"),
    "qwen3-8b": _mcore("qwen3/qwen3_8b.yaml"),
    "qwen3-14b": _mcore("qwen3/qwen3_14b.yaml"),
    "qwen3-30b-a3b": _mcore("qwen3/qwen3_30b_a3b.yaml"),
    "qwen3-32b": _mcore("qwen3/qwen3_32b.yaml"),
    "qwen3-235b-a22b": _mcore("qwen3/qwen3_235b_a22b.yaml"),
    "qwen3-480b-a35b": _mcore("qwen3/qwen3_480b_a35b.yaml"),
    "qwen3-coder-30b-a3b": _mcore("qwen3/qwen3_coder_30b_a3b.yaml"),

    # qwen3-next-80b-a3b
    "qwen3-next-80b-a3b": _mcore("qwen3_next/qwen3_next_80b_a3b.yaml"),

    # Kimi K3 multimodal model
    "kimi-k3": _mcore("kimi_k3/kimi_k3.yaml"),

    # qwen3.5
    "qwen3.5-0.8b": _mcore("qwen3.5/qwen3_5_0_8b.yaml"),
    "qwen3.5-2b": _mcore("qwen3.5/qwen3_5_2b.yaml"),
    "qwen3.5-4b": _mcore("qwen3.5/qwen3_5_4b.yaml"),
    "qwen3.5-9b": _mcore("qwen3.5/qwen3_5_9b.yaml"),
    "qwen3.5-27b": _mcore("qwen3.5/qwen3_5_27b.yaml"),
    "qwen3.5-35b-a3b": _mcore("qwen3.5/qwen3_5_35b_a3b.yaml"),
    "qwen3.5-122b-a10b": _mcore("qwen3.5/qwen3_5_122b_a10b.yaml"),
    "qwen3.5-397b-a17b": _mcore("qwen3.5/qwen3_5_397b_a17b.yaml"),

    # qwen3.6
    "qwen3.6-27b": _mcore("qwen3.6/qwen3_6_27b.yaml"),
    "qwen3.6-35b-a3b": _mcore("qwen3.6/qwen3_6_35b_a3b.yaml"),

    # qwen3.8
    "qwen3.8-27b": _mcore("qwen3.8/qwen3_8_27b.yaml"),

    # kimi-k2.x
    "kimi-k2.5": _mcore("kimi_k2.5/kimi_k2_5.yaml"),
    "kimi-k2.6": _mcore("kimi_k2.6/kimi_k2_6.yaml"),

    # qwen2.5-vl
    "qwen2.5-vl-3b": _mcore("qwen2.5vl/qwen2_5_vl_3b.yaml"),
    "qwen2.5-vl-3b-lora": _mcore("qwen2.5vl/qwen2_5_vl_3b_lora.yaml"),
    "qwen2.5-vl-7b": _mcore("qwen2.5vl/qwen2_5_vl_7b.yaml"),
    "qwen2.5-vl-32b": _mcore("qwen2.5vl/qwen2_5_vl_32b.yaml"),
    "qwen2.5-vl-72b": _mcore("qwen2.5vl/qwen2_5_vl_72b.yaml"),

    # internvl 2.5
    "internvl2.5-8b": _mcore("internvl2.5/internvl2_5_8b.yaml"),
    "internvl2.5-26b": _mcore("internvl2.5/internvl2_5_26b.yaml"),
    "internvl2.5-38b": _mcore("internvl2.5/internvl2_5_38b.yaml"),
    "internvl2.5-78b": _mcore("internvl2.5/internvl2_5_78b.yaml"),

    # internvl 3.5
    "internvl3.5-8b": _mcore("internvl3.5/internvl3_5_8b.yaml"),
    "internvl3.5-14b": _mcore("internvl3.5/internvl3_5_14b.yaml"),
    "internvl3.5-30b-a3b": _mcore("internvl3.5/internvl3_5_30b_a3b.yaml"),
    "internvl3.5-38b": _mcore("internvl3.5/internvl3_5_38b.yaml"),
    "internvl3.5-241b-a28b": _mcore("internvl3.5/internvl3_5_241b_a28b.yaml"),

    # llavaov 1.5
    "llava-onevision-1.5-4b": _mcore("llava_onevision/llava_onevision_1_5_4b.yaml"),

    # qwen3-vl
    "qwen3-vl-30b-a3b": _mcore("qwen3_vl/qwen3_vl_30b_a3b.yaml"),
    "qwen3-vl-235b-a22b": _mcore("qwen3_vl/qwen3_vl_235b_a22b.yaml"),

    # minicpm-v
    "minicpm-v-4.6": _mcore("minicpm_v_4_6/minicpm_v_4_6.yaml"),

    # wan
    "wan2-1-i2v": _mcore("wan/wan2_1_i2v.yaml"),
    "wan2-2-i2v": _mcore("wan/wan2_2_i2v.yaml"),

    # qwen image
    "qwen-image-edit-2511": _mcore("qwen_image/qwen_image_edit_2511.yaml"),

    # mimo
    "mimo": _mcore("mimo/mimo_7b.yaml"),

    # minimax
    "minimax2.1-230b": _mcore("minimax/minimax_m2_1.yaml"),
    "minimax2.5-230b": _mcore("minimax/minimax_m2_5.yaml"),
    "minimax2.7-230b": _mcore("minimax/minimax_m2_7.yaml"),

    # ernie4.5-vl
    "ernie4.5-28b-a3b-base": _mcore("ernie4_5_vl/ernie4_5_28b_a3b_base.yaml"),
    "ernie4.5-vl-28b-a3b": _mcore("ernie4_5_vl/ernie4_5_vl_28b_a3b.yaml"),
    "glm5": _mcore("glm5/glm5.yaml"),
    "glm5.2": _mcore("glm5.2/glm5_2.yaml"),
}


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
    spec = TORCH_CONFIGS.get(model_name.lower().replace("-", "_"))
    if spec is None:
        spec = MCORE_CONFIGS.get(model_name.lower())
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}")
    if engine is not None and spec.engine != engine:
        raise ValueError(f"Model {model_name} supports {spec.engine}, not {engine}")
    if not spec.config_file.is_file():
        raise ValueError(f"Model config does not exist: {spec.config_file}")
    return spec
