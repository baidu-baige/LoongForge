# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""constants"""

from typing import List

IGNORE_INDEX = -100

######### dataset ########
DEFAULT_DATASET_NAME = "default"

DEFAULT_DATASET_CONFIG = "sft_dataset_config.yaml"

SFT_SUPPORT_DATA_TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


class SFTDataFormats(object):
    """sft data formats"""

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"


class DataRoles(object):
    """data roles"""

    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"
    SYSTEM = "system"


class Placeholder(object):
    """Placeholders"""

    IMAGE = "<image>"
    VIDEO = "<video>"


######## training args ########
class TrainingPhase(object):
    """Training phase"""

    PRETRAIN = "pretrain"
    SFT = "sft"


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
    DEEPSEEK = "deepseek"
    INTERNLM2_5 = "internlm2.5"
    MINIMAX = "minimax"
    MIMO = "mimo"
    GLM = "glm"


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
    KIMI_K2_5 = "kimi_k2_5"


class CustomModelFamilies(_BaseFamilies):
    """User defined Custom Vision model families"""
    WAN2_2_I2V = "wan2_2_i2v"


class VisionLanguageActionModelFamilies(_BaseFamilies):
    """Vision language action model families"""
    PI05 = "pi05"
    GROOT_N1_6 = "groot_n1_6"


def get_all_model_families() -> List[str]:
    """
    Get all model families defined in the constants file dynamically.
    Returns a flattened list of all string names from all subclasses of _BaseFamilies.
    """
    all_families = []

    for family_class in _BaseFamilies.__subclasses__():
        all_families.extend(family_class.names())
        
    return all_families
