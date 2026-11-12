# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model catalog and engine-aware configuration bindings."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import List, Type


_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"


class _BaseFamilies(object):
    @classmethod
    def names(cls) -> List[str]:
        """Return all string-valued family names defined on the class."""
        return [
            value
            for name, value in vars(cls).items()
            if isinstance(value, str) and not name.startswith("__")
        ]


class LanguageModelFamilies(_BaseFamilies):
    """Language model families."""

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
    """Vision-language model families."""

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
    """Diffusion model families."""

    WAN2_1_I2V = "wan2_1_i2v"
    WAN2_2_I2V = "wan2_2_i2v"
    QWEN_IMAGE = "qwen_image"


class EmbodiedModelFamilies(_BaseFamilies):
    """Embodied model families."""

    PI05 = "pi05"
    GR00T_N1_6 = "Gr00tN1d6"
    GR00T_N1_7 = "Gr00tN1d7"
    XVLA = "xvla"
    VLDA = "vlda"
    FASTWAM = "fastwam"
    LINGBOT_VA = "lingbot_va"
    LINGBOT_VLA_V2 = "lingbot_vla_v2"
    COSMOS3 = "cosmos3"
    DREAMZERO = "dreamzero"
    WALL_OSS_0_5 = "wall_oss_0_5"
    GIGA_BRAIN07 = "giga_brain07"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved engine and configuration for one public model name."""

    engine: str
    yaml_file: str
    model_config_type: str | None = None
    data_config_type: str | None = None


@dataclass(frozen=True)
class ModelSchema:
    """Resolved typed Torch configuration classes for one model name."""

    model_config_cls: Type
    data_config_cls: Type


# Torch entries include the YAML path and typed model/data config paths.
# The outer key selects the execution engine; the next key is the model family.
# The same family may appear under both engines when implementations diverge.
MODEL_CONFIG_REGISTRY = {
    "megatron": {
        LanguageModelFamilies.DEEPSEEK: {
            "deepseek-v2": "deepseek2/deepseek_v2.yaml",
            "deepseek-v2-lite": "deepseek2/deepseek_v2_lite.yaml",
            "deepseek-v3": "deepseek3/deepseek_v3.yaml",
            "deepseek-v3.2-sparse": "deepseek3/deepseek_v3_2_sparse.yaml",
        },
        LanguageModelFamilies.DEEPSEEK_V4: {
            "deepseek-v4-flash": "deepseek4/deepseek_v4_flash_base.yaml",
            "deepseek-v4-flash-lite": "deepseek4/deepseek_v4_flash_lite.yaml",
            "deepseek-v4-flash-lite-2l": "deepseek4/deepseek_v4_flash_lite_2l.yaml",
            "deepseek-v4-flash-lite-4l": "deepseek4/deepseek_v4_flash_lite_4l.yaml",
            "deepseek-v4-flash-lite-6l": "deepseek4/deepseek_v4_flash_lite_6l.yaml",
            "deepseek-v4-pro": "deepseek4/deepseek_v4_pro_base.yaml",
        },
        LanguageModelFamilies.INTERNLM2_5: {
            "internlm2.5-8b": "internlm2.5/internlm2_5_8b.yaml",
            "internlm2.5-20b": "internlm2.5/internlm2_5_20b.yaml",
        },
        LanguageModelFamilies.LLAMA: {
            "llama2-7b": "llama2/llama2_7b.yaml",
            "llama2-13b": "llama2/llama2_13b.yaml",
            "llama2-70b": "llama2/llama2_70b.yaml",
            "llama3-8b": "llama3/llama3_8b.yaml",
            "llama3-70b": "llama3/llama3_70b.yaml",
            "llama3.1-8b": "llama3/llama3_1_8b.yaml",
            "llama3.1-70b": "llama3/llama3_1_70b.yaml",
            "llama3.1-405b": "llama3/llama3_1_405b.yaml",
        },
        LanguageModelFamilies.QWEN: {
            "qwen-1.8b": "qwen/qwen_1_8b.yaml",
            "qwen-7b": "qwen/qwen_7b.yaml",
            "qwen-14b": "qwen/qwen_14b.yaml",
            "qwen-72b": "qwen/qwen_72b.yaml",
            "qwen1.5-0.5b": "qwen/qwen1_5_0_5b.yaml",
            "qwen1.5-1.8b": "qwen/qwen1_5_1_8b.yaml",
            "qwen1.5-4b": "qwen/qwen1_5_4b.yaml",
            "qwen1.5-7b": "qwen/qwen1_5_7b.yaml",
            "qwen1.5-14b": "qwen/qwen1_5_14b.yaml",
            "qwen1.5-32b": "qwen/qwen1_5_32b.yaml",
            "qwen1.5-72b": "qwen/qwen1_5_72b.yaml",
            "qwen2-0.5b": "qwen2/qwen2_0_5b.yaml",
            "qwen2-1.5b": "qwen2/qwen2_1_5b.yaml",
            "qwen2-7b": "qwen2/qwen2_7b.yaml",
            "qwen2-72b": "qwen2/qwen2_72b.yaml",
            "qwen2.5-0.5b": "qwen2.5/qwen2_5_0_5b.yaml",
            "qwen2.5-1.5b": "qwen2.5/qwen2_5_1_5b.yaml",
            "qwen2.5-3b": "qwen2.5/qwen2_5_3b.yaml",
            "qwen2.5-7b": "qwen2.5/qwen2_5_7b.yaml",
            "qwen2.5-14b": "qwen2.5/qwen2_5_14b.yaml",
            "qwen2.5-32b": "qwen2.5/qwen2_5_32b.yaml",
            "qwen2.5-72b": "qwen2.5/qwen2_5_72b.yaml",
            "qwen3-0.6b": "qwen3/qwen3_0_6b.yaml",
            "qwen3-1.7b": "qwen3/qwen3_1_7b.yaml",
            "qwen3-4b": "qwen3/qwen3_4b.yaml",
            "qwen3-8b": "qwen3/qwen3_8b.yaml",
            "qwen3-14b": "qwen3/qwen3_14b.yaml",
            "qwen3-30b-a3b": "qwen3/qwen3_30b_a3b.yaml",
            "qwen3-32b": "qwen3/qwen3_32b.yaml",
            "qwen3-235b-a22b": "qwen3/qwen3_235b_a22b.yaml",
            "qwen3-480b-a35b": "qwen3/qwen3_480b_a35b.yaml",
            "qwen3-coder-30b-a3b": "qwen3/qwen3_coder_30b_a3b.yaml",
        },
        LanguageModelFamilies.QWEN3_NEXT: {
            "qwen3-next-80b-a3b": "qwen3_next/qwen3_next_80b_a3b.yaml",
        },
        VisionLanguageModelFamilies.KIMI_K3: {
            "kimi-k3": "kimi_k3/kimi_k3.yaml",
        },
        VisionLanguageModelFamilies.QWEN3_5: {
            "qwen3.5-0.8b": "qwen3.5/qwen3_5_0_8b.yaml",
            "qwen3.5-2b": "qwen3.5/qwen3_5_2b.yaml",
            "qwen3.5-4b": "qwen3.5/qwen3_5_4b.yaml",
            "qwen3.5-9b": "qwen3.5/qwen3_5_9b.yaml",
            "qwen3.5-27b": "qwen3.5/qwen3_5_27b.yaml",
            "qwen3.5-35b-a3b": "qwen3.5/qwen3_5_35b_a3b.yaml",
            "qwen3.5-122b-a10b": "qwen3.5/qwen3_5_122b_a10b.yaml",
            "qwen3.5-397b-a17b": "qwen3.5/qwen3_5_397b_a17b.yaml",
            "qwen3.6-27b": "qwen3.6/qwen3_6_27b.yaml",
            "qwen3.6-35b-a3b": "qwen3.6/qwen3_6_35b_a3b.yaml",
            "qwen3.8-27b": "qwen3.8/qwen3_8_27b.yaml",
        },
        VisionLanguageModelFamilies.KIMI_K2_5: {
            "kimi-k2.5": "kimi_k2.5/kimi_k2_5.yaml",
        },
        VisionLanguageModelFamilies.KIMI_K2_6: {
            "kimi-k2.6": "kimi_k2.6/kimi_k2_6.yaml",
        },
        VisionLanguageModelFamilies.QWEN2_5_VL: {
            "qwen2.5-vl-3b": "qwen2.5vl/qwen2_5_vl_3b.yaml",
            "qwen2.5-vl-3b-lora": "qwen2.5vl/qwen2_5_vl_3b_lora.yaml",
            "qwen2.5-vl-7b": "qwen2.5vl/qwen2_5_vl_7b.yaml",
            "qwen2.5-vl-32b": "qwen2.5vl/qwen2_5_vl_32b.yaml",
            "qwen2.5-vl-72b": "qwen2.5vl/qwen2_5_vl_72b.yaml",
        },
        VisionLanguageModelFamilies.INTERN_VL: {
            "internvl2.5-8b": "internvl2.5/internvl2_5_8b.yaml",
            "internvl2.5-26b": "internvl2.5/internvl2_5_26b.yaml",
            "internvl2.5-38b": "internvl2.5/internvl2_5_38b.yaml",
            "internvl2.5-78b": "internvl2.5/internvl2_5_78b.yaml",
            "internvl3.5-8b": "internvl3.5/internvl3_5_8b.yaml",
            "internvl3.5-14b": "internvl3.5/internvl3_5_14b.yaml",
            "internvl3.5-30b-a3b": "internvl3.5/internvl3_5_30b_a3b.yaml",
            "internvl3.5-38b": "internvl3.5/internvl3_5_38b.yaml",
            "internvl3.5-241b-a28b": "internvl3.5/internvl3_5_241b_a28b.yaml",
        },
        VisionLanguageModelFamilies.LLAVA_OV_1_5: {
            "llava-onevision-1.5-4b": "llava_onevision/llava_onevision_1_5_4b.yaml",
        },
        VisionLanguageModelFamilies.QWEN3_VL: {
            "qwen3-vl-30b-a3b": "qwen3_vl/qwen3_vl_30b_a3b.yaml",
            "qwen3-vl-235b-a22b": "qwen3_vl/qwen3_vl_235b_a22b.yaml",
        },
        VisionLanguageModelFamilies.MINICPM_V_4_6: {
            "minicpm-v-4.6": "minicpm_v_4_6/minicpm_v_4_6.yaml",
        },
        DiffusionModelFamilies.WAN2_1_I2V: {
            "wan2-1-i2v": "wan/wan2_1_i2v.yaml",
        },
        DiffusionModelFamilies.WAN2_2_I2V: {
            "wan2-2-i2v": "wan/wan2_2_i2v.yaml",
        },
        DiffusionModelFamilies.QWEN_IMAGE: {
            "qwen-image-edit-2511": "qwen_image/qwen_image_edit_2511.yaml",
        },
        LanguageModelFamilies.MIMO: {
            "mimo": "mimo/mimo_7b.yaml",
        },
        LanguageModelFamilies.MINIMAX: {
            "minimax2.1-230b": "minimax/minimax_m2_1.yaml",
            "minimax2.5-230b": "minimax/minimax_m2_5.yaml",
            "minimax2.7-230b": "minimax/minimax_m2_7.yaml",
        },
        LanguageModelFamilies.GLM: {
            "glm5": "glm5/glm5.yaml",
            "glm5.2": "glm5.2/glm5_2.yaml",
        },
        LanguageModelFamilies.GLM5_NEXT: {
            "glm5.3-flash": "glm5_next/glm5.3_flash.yaml",
        },
    },
    "torch": {
        EmbodiedModelFamilies.LINGBOT_VA: {
            "lingbot_va_robotwin": (
                "lingbot_va/lingbot_va_robotwin.yaml",
                "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
                (
                    "loongforge.data.embodied.lingbot_va.transforms.data_configuration_lingbot_va."
                    "LingBotVADataConfig"
                ),
            ),
            "lingbot_va_libero": (
                "lingbot_va/lingbot_va_libero.yaml",
                "loongforge.models.embodied.lingbot_va.model_configuration_lingbot_va.LingBotVAModelConfig",
                (
                    "loongforge.data.embodied.lingbot_va.transforms.data_configuration_lingbot_va."
                    "LingBotVADataConfig"
                ),
            ),
        },
        EmbodiedModelFamilies.LINGBOT_VLA_V2: {
            "lingbot_vla_v2": (
                "lingbot_vla_v2/lingbot_vla_v2.yaml",
                (
                    "loongforge.models.embodied.lingbot_vla_v2.model_configuration_lingbot_vla_v2."
                    "LingbotVLAV2ModelConfig"
                ),
                (
                    "loongforge.data.embodied.lingbot_vla_v2.transforms.data_configuration_lingbot_vla_v2."
                    "LingbotVLAV2DataConfig"
                ),
            ),
        },
        EmbodiedModelFamilies.PI05: {
            "pi05": (
                "pi05/pi05.yaml",
                "loongforge.models.embodied.pi05.model_configuration_pi05.Pi05ModelConfig",
                "loongforge.data.embodied.pi05.transforms.data_configuration_pi05.Pi05DataConfig",
            ),
        },
        EmbodiedModelFamilies.VLDA: {
            "vlda": (
                "vlda/vlda.yaml",
                "loongforge.models.embodied.vlda.model_configuration_vlda.VLDAModelConfig",
                "loongforge.data.embodied.vlda.transforms.data_configuration_vlda.VLDADataConfig",
            ),
        },
        EmbodiedModelFamilies.GR00T_N1_6: {
            "gr00t_n1_6": (
                "gr00t_n1_6/gr00t_n1_6.yaml",
                "loongforge.models.embodied.gr00t_n1_6.model_configuration_gr00t_n1_6.GrootN1d6ModelConfig",
                (
                    "loongforge.data.embodied.gr00t_n1_6.transforms.data_configuration_gr00t_n1_6."
                    "GrootN1d6DataConfig"
                ),
            ),
        },
        EmbodiedModelFamilies.XVLA: {
            "xvla": (
                "xvla/xvla.yaml",
                "loongforge.models.embodied.xvla.model_configuration_xvla.XvlaModelConfig",
                "loongforge.data.embodied.xvla.transforms.data_configuration_xvla.XvlaDataConfig",
            ),
        },
        EmbodiedModelFamilies.FASTWAM: {
            "fastwam": (
                "fastwam/fastwam.yaml",
                "loongforge.models.embodied.fastwam.modeling_configuration_fastwam.FastWAMModelConfig",
                "loongforge.data.embodied.fastwam.transforms.data_configuration_fastwam.FastWAMDataConfig",
            ),
        },
        EmbodiedModelFamilies.GR00T_N1_7: {
            "gr00t_n1_7": (
                "gr00t_n1_7/gr00t_n1_7.yaml",
                "loongforge.models.embodied.gr00t_n1_7.model_configuration_gr00t_n1_7.GrootN1d7Config",
                (
                    "loongforge.data.embodied.gr00t_n1_7.transforms.data_configuration_gr00t_n1_7."
                    "GrootN1d7DataConfig"
                ),
            ),
        },
        EmbodiedModelFamilies.COSMOS3: {
            "cosmos3_nano": (
                "cosmos3/nano.yaml",
                "loongforge.models.embodied.cosmos3.modeling_configuration_cosmos3.Cosmos3ModelConfig",
                "loongforge.data.embodied.cosmos3.data_configuration_cosmos3.Cosmos3DroidConfig",
            ),
        },
        EmbodiedModelFamilies.DREAMZERO: {
            "dreamzero_lora_wan22_5b": (
                "dreamzero/dreamzero_wan22_5b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_full_wan22_5b": (
                "dreamzero/dreamzero_wan22_5b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_lora_wan21_14b": (
                "dreamzero/dreamzero_wan21_14b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_full_wan21_14b": (
                "dreamzero/dreamzero_wan21_14b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_libero_wan22_5b": (
                "dreamzero/dreamzero_libero_wan22_5b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_agibot_wan21_14b": (
                "dreamzero/dreamzero_agibot_wan21_14b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
            "dreamzero_yam_wan21_14b": (
                "dreamzero/dreamzero_yam_wan21_14b.yaml",
                "loongforge.models.embodied.dreamzero.model_configuration_dreamzero.DreamZeroConfig",
                "loongforge.data.embodied.dreamzero.transforms.data_configuration_dreamzero.DreamZeroDataConfig",
            ),
        },
        EmbodiedModelFamilies.WALL_OSS_0_5: {
            "wall_oss_0_5": (
                "wall_oss_0_5/wall_oss_0_5.yaml",
                "loongforge.models.embodied.wall_oss_0_5.model_configuration_wall_oss_0_5.WallOss05ModelConfig",
                (
                    "loongforge.data.embodied.wall_oss_0_5.transforms.data_configuration_wall_oss_0_5."
                    "WallOss05DataConfig"
                ),
            ),
        },
        EmbodiedModelFamilies.GIGA_BRAIN07: {
            "giga_brain07": (
                "giga_brain07/giga_brain07.yaml",
                (
                    "loongforge.models.embodied.giga_brain07.model_configuration_giga_brain07."
                    "GigaBrain07ModelConfig"
                ),
                (
                    "loongforge.data.embodied.giga_brain07.transforms.data_configuration_giga_brain07."
                    "GigaBrain07DataConfig"
                ),
            ),
        },
    },
}


def _iter_model_specs():
    for engine, families in MODEL_CONFIG_REGISTRY.items():
        for models in families.values():
            for model_name, config in models.items():
                configs = config if engine == "torch" else (config,)
                yield model_name, ModelSpec(engine, *configs)


def get_model_spec(model_name: str) -> ModelSpec:
    """Resolve a public model name to its engine and config."""
    name = model_name.lower()
    for registered_name, spec in _iter_model_specs():
        lookup_name = name.replace("-", "_") if spec.engine == "torch" else name
        if registered_name == lookup_name:
            return spec
    available = sorted(registered_name for registered_name, _ in _iter_model_specs())
    raise KeyError(
        f"Unknown model_name {model_name!r}. Register it in "
        f"{__name__}.MODEL_CONFIG_REGISTRY. Available: {available}"
    )


def _config_file(spec: ModelSpec) -> Path:
    return _CONFIGS_DIR / spec.yaml_file


def get_config_from_model_name(model_name: str):
    """Return the Hydra config directory and name for a Megatron model."""
    spec = get_model_spec(model_name)
    if spec.engine != "megatron":
        raise ValueError(f"Model {model_name!r} does not use Megatron configs")
    config_file = _config_file(spec)
    return str(config_file.parent), config_file.stem


def get_config_path(model_name: str) -> str:
    """Return the existing default YAML path for a model name."""
    config_file = _config_file(get_model_spec(model_name))
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return str(config_file)


def _load_type(path: str) -> Type:
    module, name = path.rsplit(".", 1)
    return getattr(import_module(module), name)


def get_model_schema(model_name: str) -> ModelSchema:
    """Resolve the typed Torch config classes for a model name."""
    spec = get_model_spec(model_name)
    if spec.engine != "torch":
        raise ValueError(f"Model {model_name!r} does not use Torch config types")
    return ModelSchema(
        _load_type(spec.model_config_type),
        _load_type(spec.data_config_type),
    )


def get_all_model_families() -> List[str]:
    """Return all family names defined in the catalog constants."""
    return [
        family
        for family_class in _BaseFamilies.__subclasses__()
        for family in family_class.names()
    ]
